import torch
import torch.nn as nn
from types import SimpleNamespace

from diffusers.models.modeling_utils import ModelMixin


class DualUNet(ModelMixin):
    """
    Routes each denoising step to either the standard UNet or RAU-Net
    depending on the current timestep relative to t_threshold (tau).

      t < t_threshold  ->  unet_std  (fine detail phase)
      t >= t_threshold ->  unet_rau  (structure phase, anti-replication)
    """

    def __init__(self, unet_std, unet_rau, t_threshold, config=None):
        super().__init__()
        self.unet_std = unet_std
        self.unet_rau = unet_rau
        self.t_threshold = t_threshold
        self.config = config or SimpleNamespace(
            in_channels=4, out_channels=4, sample_size=96,
            attention_head_dim=[5, 10, 20, 20],
        )

    @classmethod
    def from_two_unets(cls, unet_std, unet_rau, t_threshold: int):
        try:
            std_cfg = dict(unet_std.config)
            rau_cfg = dict(unet_rau.config)
            merged = {k: v for k, v in std_cfg.items() if k in rau_cfg and std_cfg[k] == rau_cfg[k]}
            config = SimpleNamespace(**merged)
        except Exception:
            config = None
        return cls(unet_std=unet_std, unet_rau=unet_rau, t_threshold=t_threshold, config=config)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        std_mask = timestep < self.t_threshold
        rau_mask = ~std_mask

        std_out = rau_out = None

        if std_mask.any():
            std_out = self.unet_std(sample[std_mask], timestep[std_mask],
                                    encoder_hidden_states[std_mask], **kwargs)
        if rau_mask.any():
            rau_out = self.unet_rau(sample[rau_mask], timestep[rau_mask],
                                    encoder_hidden_states[rau_mask], **kwargs)

        if std_out is None:
            return rau_out
        if rau_out is None:
            return std_out

        # Merge outputs when batch has mixed timesteps
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        full = torch.zeros_like(sample)
        full[std_mask] = std_out.sample
        full[rau_mask] = rau_out.sample
        return UNet2DConditionOutput(sample=full)
