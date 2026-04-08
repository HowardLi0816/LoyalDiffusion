import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Union

from diffusers import UNet2DConditionModel
from diffusers.utils import BaseOutput


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.Tensor = None


class RAUNet(UNet2DConditionModel):
    """
    Replication-Aware U-Net (RAU-Net).

    Inserts Conv 3x3 blocks on selected skip connections before they are
    passed to the decoder. The paper uses SC3 and SC4 (block indices 2 and 3).
    """

    def __init__(self, sc_block_indices: List[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sc_block_indices is None:
            sc_block_indices = [2, 3]
        self.sc_block_indices = sc_block_indices

        block_out_channels = kwargs.get("block_out_channels", [320, 640, 1280, 1280])

        # Build per-skip-connection conv modules.
        # Index 0 is the conv_in output; remaining indices follow each resnet/downsampler.
        if -1 in sc_block_indices:
            first_conv = nn.Conv2d(block_out_channels[0], block_out_channels[0], 3, padding=1)
        else:
            first_conv = nn.Identity()
        self.skip_conv_modules = nn.ModuleList([first_conv])

        for i, down_block in enumerate(self.down_blocks):
            if i in sc_block_indices:
                for resnet in down_block.resnets:
                    ch = resnet.out_channels
                    self.skip_conv_modules.append(nn.Conv2d(ch, ch, 3, padding=1))
                if down_block.downsamplers is not None:
                    ch = block_out_channels[i]
                    self.skip_conv_modules.append(nn.Conv2d(ch, ch, 3, padding=1))
            else:
                for _ in down_block.resnets:
                    self.skip_conv_modules.append(nn.Identity())
                if down_block.downsamplers is not None:
                    self.skip_conv_modules.append(nn.Identity())

    @classmethod
    def from_config(cls, config: dict, sc_block_indices: List[int] = None, **kwargs):
        clean = {k: v for k, v in config.items() if not k.startswith("_")}
        return cls(sc_block_indices=sc_block_indices or [2, 3], **clean, **kwargs)

    def load_sd21_weights(self, weight_path: str):
        state_dict = torch.load(weight_path, map_location="cpu")
        missing, _ = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[RAUNet] {len(missing)} keys not in checkpoint (new conv layers); initialized randomly.")
        return self

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        class_labels = kwargs.get("class_labels", None)
        timestep_cond = kwargs.get("timestep_cond", None)
        attention_mask = kwargs.get("attention_mask", None)
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", None)
        added_cond_kwargs = kwargs.get("added_cond_kwargs", None)
        down_block_additional_residuals = kwargs.get("down_block_additional_residuals", None)
        mid_block_additional_residual = kwargs.get("mid_block_additional_residual", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        return_dict = kwargs.get("return_dict", True)

        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = any(s % default_overall_up_factor != 0 for s in sample.shape[-2:])
        upsample_size = None

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            dtype = torch.float32 if (is_mps and isinstance(timestep, float)) else (
                torch.int32 if is_mps else torch.int64)
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels required when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels).to(dtype=sample.dtype)
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            emb = torch.cat([emb, class_emb], dim=-1) if self.config.class_embeddings_concat else emb + class_emb

        if self.config.addition_embed_type == "text_time" and added_cond_kwargs is not None:
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten()).reshape((text_embeds.shape[0], -1))
            aug_emb = self.add_embedding(torch.concat([text_embeds, time_embeds], dim=-1).to(emb.dtype))

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                sample, res_samples = down_block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_samples = ()
            for s, r in zip(down_block_res_samples, down_block_additional_residuals):
                new_samples += (s + r,)
            down_block_res_samples = new_samples

        # Apply conv transform on selected skip connections
        down_block_res_samples = tuple(
            conv(feat) for conv, feat in zip(self.skip_conv_modules, down_block_res_samples)
        )

        if self.mid_block is not None:
            sample = self.mid_block(
                sample, emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        for i, up_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample, temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = up_block(hidden_states=sample, temb=emb,
                                  res_hidden_states_tuple=res_samples, upsample_size=upsample_size)

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        return UNet2DConditionOutput(sample=sample)
