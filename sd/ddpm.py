import torch
import numpy as np


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) 噪声调度器。

    负责管理扩散过程的噪声调度表，并提供前向加噪和反向去噪的采样方法。
    """

    def __init__(
        self,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        generator: torch.Generator = None
    ):
        """
        初始化 DDPM 调度器。

        参数:
            num_training_steps: 训练时的总扩散步数 (默认 1000)
            beta_start: beta 调度起始值 (SD v1.5 标准值 0.00085)
            beta_end: beta 调度终止值 (SD v1.5 标准值 0.012)
            generator: 随机数生成器，用于可复现采样
        """
        # 使用 "scaled linear" beta 调度策略
        # 在 sqrt(beta) 空间线性插值，这是 SD v1.5 的标准做法
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # α̅_t = ∏(α_i)

        self.num_training_steps = num_training_steps
        self.generator = generator

        # 推理时使用的时间步序列（将在 set_inference_timesteps 中设置）
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        """
        设置推理时的采样步数。

        从训练的 1000 步中均匀选取子集，用于加速推理。

        参数:
            num_inference_steps: 推理步数 (典型值 20-50)
        """
        step_ratio = self.num_training_steps // num_inference_steps
        # 从 999 开始，每隔 step_ratio 取一个时间步，降序排列
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """获取前一个时间步 t-1"""
        prev_t = timestep - self.num_training_steps // len(self.timesteps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        计算时间步 t 的方差。

        DDPM 论文中的 σ_t^2 = β_t
        """
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # DDPM 方差公式
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 防止数值不稳定
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        前向扩散：给干净的 latent 加噪到指定时间步。

        用于 img2img 场景，将输入图像编码后的 latent 加噪。

        公式: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

        参数:
            original_samples: 原始干净样本, shape (B, C, H, W)
            timesteps: 目标时间步, shape (B,) 或标量

        返回:
            加噪后的样本, shape 同 original_samples
        """
        # 确保 alpha_cumprod 在正确的设备上
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # 获取对应时间步的 α̅_t
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 生成噪声
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype
        )

        # 应用加噪公式
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        反向去噪：执行一步 DDPM 去噪采样。

        参数:
            model_output: U-Net 预测的噪声 ε_θ, shape (B, C, H, W)
            timestep: 当前时间步 t
            sample: 当前含噪样本 x_t, shape (B, C, H, W)

        返回:
            去噪一步后的样本 x_{t-1}, shape 同 sample
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 获取 α_t, α̅_t, α̅_{t-1}
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 预测原始样本 x_0
        # x_0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 计算 x_{t-1} 的系数
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # 计算 x_{t-1}
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 添加噪声（最后一步 t=0 时不加噪声）
        variance = 0
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
