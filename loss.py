import torch
import torchmetrics.image
from lpips import LPIPS
from kornia import color

class ImageReconstructionLoss(torch.nn.Module):
    def __init__(self, recon_type='yuv', recon_weight=1.0, perceptual_weight=1.0, logvar_init=0.0, ramp=100000, max_weight_ratio=2.0):
        super().__init__()
        assert recon_type in ['rgb', 'yuv']
        self.recon_type = recon_type
        if recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1, 100, 100]).unsqueeze(1).float())  # 强化色彩通道
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight

        self.logvar = torch.nn.Parameter(torch.ones(size=()) * logvar_init)
        self.perceptual_loss = LPIPS().eval().cpu()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).cpu()

        self.ramp = ramp
        self.max_weight = max_weight_ratio - 1  # 因为 image_weight = 1 + something
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))  # 不激活前不会变动

    def activate_ramp(self, global_step):
        if not self.ramp_on:
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print(f"[TRAINING] Ramp activated at step {global_step}")

    def compute_recon_loss(self, x, x_recon):
        if self.recon_type == 'rgb':
            l1 = self.l1_loss(x, x_recon)
            ssim_loss = 1.0 - self.ssim(x, x_recon)
            return l1 + ssim_loss
        elif self.recon_type == 'yuv':
            x_yuv = color.rgb_to_yuv(x)
            x_recon_yuv = color.rgb_to_yuv(x_recon)
            yuv_loss = torch.mean((x_yuv - x_recon_yuv) ** 2, dim=[2, 3])  # [B, C]
            yuv_scaled = torch.mm(yuv_loss, self.yuv_scales.to(yuv_loss.device)).squeeze(1)
            l1 = self.l1_loss(x, x_recon)
            ssim_loss = 1.0 - self.ssim(x, x_recon)
            return yuv_scaled.mean() + l1 + ssim_loss

    def forward(self, x, x_recon, global_step):
        recon_loss = self.compute_recon_loss(x, x_recon) * self.recon_weight

        # Perceptual loss
        perceptual = self.perceptual_loss(x_recon, x).mean()

        # logvar loss scaling
        image_loss = (recon_loss + perceptual * self.perceptual_weight)
        image_loss_scaled = image_loss / torch.exp(self.logvar) + self.logvar

        # dynamic ramp weight
        if global_step >= self.step0.item():
            weight = 1 + min(self.max_weight, self.max_weight * (global_step - self.step0.item()) / self.ramp)
        else:
            weight = 1.0  # 初始阶段未激活 ramp

        final_loss = image_loss_scaled * weight / (weight + 1)  # 可视为 image_weight / total_weight

        return final_loss, {
            'recon_loss': recon_loss.item(),
            'perceptual_loss': perceptual.item(),
            'final_loss': final_loss.item(),
            'logvar': self.logvar.item(),
            'image_weight': weight
        }
