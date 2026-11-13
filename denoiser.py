import torch
import torch.nn as nn
from model_pixeldit import PixelDiT_models
from model_deco import DeCo_models
from repa_encoder import DINOv2
from util.no_grad import no_grad
import torch.nn.functional as F
import math
from torch import Tensor

class BaseScheduler:
    def alpha(self, t) -> Tensor:
        ...
    def sigma(self, t) -> Tensor:
        ...

    def dalpha(self, t) -> Tensor:
        ...
    def dsigma(self, t) -> Tensor:
        ...

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)

    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t)*self.sigma(t)

    def drift_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dalpha/(alpha + 1e-6)

    def diffuse_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dsigma*sigma - dalpha/(alpha + 1e-6)*sigma**2

    def w(self, t):
        return self.sigma(t)

class LinearScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return (t).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return (1-t).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        if args.model_type == 'PixelDiT':
            self.net = PixelDiT_models[args.model]()

        elif args.model_type == 'DeCo':
            self.net = DeCo_models[args.model]()
        
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        self.scheduler = LinearScheduler()

        # irepa
        self.align_layer = args.repa_depth
        self.encoder = DINOv2(base_patch_size=16) # or 32
        no_grad(self.encoder)
        self.proj = nn.Conv2d(args.model_dim, 768, kernel_size=3, stride=1, padding=1)
        self.repa_weight = args.repa_weight
        
    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def spatial_zscore(self, feat: torch.Tensor, alpha: float = 0.8, eps: float = 1e-6) -> torch.Tensor:
        """
        Z-score normalization along spatial dimension.

        Args:
            feat: (B, T, D) patch tokens
            alpha: scaling factor for mean subtraction (default 1.0)
            eps: small constant for numerical stability

        Returns:
            Normalized features (B, T, D)
        """
        mean = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True)
        return (feat - alpha * mean) / (std + eps)


    def forward(self, x, labels, epoch=None):
        # repa
        raw_images = (x+1)/2
        with torch.no_grad():
            dst_feature = self.encoder(raw_images)
        src_feature = []
        def forward_hook(net, input, output):
            feature = output
            src_feature.append(feature)
        handle = self.net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        z = alpha * x + e * sigma
        v = dalpha * x + dsigma * e
        v_pred = self.net(z, t.flatten(), labels_dropped)
        handle.remove()

        # l2 loss
        fm_loss = (v - v_pred) ** 2
        fm_loss = fm_loss.mean(dim=(1, 2, 3)).mean()

        # repa loss
        src_feature = src_feature[0]
        B,T,D = src_feature.shape
        H = W = int(math.isqrt(T))
        src_feature = src_feature.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        src_feature = self.proj(src_feature)                  # [B, z_dim, H, W]
        src_feature = src_feature.permute(0, 2, 3, 1).contiguous()         # [B, H, W, z_dim]
        src_feature = src_feature.reshape(B, T, -1)
        dst_feature = self.spatial_zscore(dst_feature)
        src_feature = F.normalize(src_feature, dim=-1) # [B,T,D]
        dst_feature = F.normalize(dst_feature, dim=-1) # [B,T,D]
        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = (1 - cos_sim).mean()
        
        return fm_loss + self.repa_weight * cos_loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        v_cond = self.net(z, t.flatten(), labels)
        # v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        v_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        # v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
