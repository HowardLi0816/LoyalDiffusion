from diffusers import UNet2DConditionModel
import math
import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union

from utils_custom_unet import deprecate
from outputs import BaseOutput

from diffusers.models.modeling_utils import ModelMixin

class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class CustomUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return output
        
    def from_pretrained(self, *args, **kwargs):
        output = super().from_pretrained(*args, **kwargs)
        return output
        
    def save_pretrained(self, *args, **kwargs):
        output = super().save_pretrained(*args, **kwargs)
        return output
        
class CustomNoCopyCropUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, oper_block_list=[0, 1, 2, 3], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oper_block_list = oper_block_list
    
    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], encoder_hidden_states: torch.Tensor, **kwargs):
    
        #sample = kwargs.get('sample')
        #timestep = kwargs.get('timestep')
        #encoder_hidden_states = kwargs.get('encoder_hidden_states')
        class_labels = kwargs.get('class_labels', None)
        timestep_cond = kwargs.get('timestep_cond', None)
        attention_mask = kwargs.get('attention_mask', None)
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs', None)
        added_cond_kwargs = kwargs.get('added_cond_kwargs', None)
        down_block_additional_residuals = kwargs.get('down_block_additional_residuals', None)
        mid_block_additional_residual = kwargs.get('mid_block_additional_residual', None)
        down_intrablock_additional_residuals = kwargs.get('down_intrablock_additional_residuals', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        return_dict = kwargs.get('return_dict', True)
        
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            
            
            #for i in range(len(res_samples)):
            #    print(res_samples[i].shape)

            down_block_res_samples += res_samples
            #print("aaaaaaaaaaaaaaaaa",len(down_block_res_samples))
            

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        
        #for i in range(len(down_block_res_samples)):
        #    print(i)
        #    print(down_block_res_samples[i].shape)
        
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            #print(len(self.up_blocks))
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            #print(len(res_samples))
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            
            #print(sample.shape)
            #print(res_samples[0].shape)
            #print(res_samples[1].shape)
            #print(res_samples[2].shape)
            
            if (len(self.up_blocks)-i-1) in self.oper_block_list:
                # form a tuple with (sample, sample, sample, ......) to match the dim and type for "res_hidden_states_tuple"
                dup_sample_as_hodden_status_tuple = ()
                for j in range(len(upsample_block.resnets)):
                    if res_samples[j].shape[1] == sample.shape[1]:
                        dup_sample = sample
                    else:
                        dup_sample = sample[:, :res_samples[j].shape[1], :, :]
                    dup_sample_as_hodden_status_tuple = dup_sample_as_hodden_status_tuple + (dup_sample,)
            else:
                dup_sample_as_hodden_status_tuple = res_samples
            
            
            ############# Change:just set res_hidden_states_tuple=dup_sample_as_hodden_status_tuple, which is the same as hidden_states. ############
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    #res_hidden_states_tuple=res_samples,
                    res_hidden_states_tuple=dup_sample_as_hodden_status_tuple,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    #res_hidden_states_tuple=res_samples,
                    res_hidden_states_tuple=dup_sample_as_hodden_status_tuple,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

class SkipUNet(nn.Module):
    def __init__(self, in_channel):
        super(SkipUNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channel//2, out_channels=in_channel//2, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=in_channel//2, out_channels=in_channel, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        x = self.silu(self.conv1(x))
        #x = self.max_pool(x)
        x = self.silu(self.up_conv(x))
        x = self.silu(self.conv2(x))
        return x
        
class MaxPoolWithIndices(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxPoolWithIndices, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # Apply max pooling
        pooled, indices = self.pool(x)
        
        # Upsample and restore original values
        restored = self.unpool(pooled, indices, output_size=x.size())
        
        return restored

class CustomAllConvCopyCropUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, conv_type="skip_unet", *args, **kwargs):
    
        # conv_type can be chosen from ["skip_unet", "single_conv", "max_pooling", "single_conv_silu"]
        
        super().__init__(*args, **kwargs)
        block_out_channels = kwargs.get('block_out_channels', None)
        if conv_type == "skip_unet":
            skip_first_conv = SkipUNet(in_channel=block_out_channels[0])
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                for resnet in downsample_block.resnets:
                    skip_in_channels = resnet.out_channels
                    skip_conv = SkipUNet(in_channel=skip_in_channels)
                    skip_conv_list.append(skip_conv)
                if downsample_block.downsamplers is not None:
                    skip_downsampler = SkipUNet(in_channel=block_out_channels[i])
                    skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
                
        elif conv_type == "single_conv":
            skip_first_conv = nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1)
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                for resnet in downsample_block.resnets:
                    skip_in_channels = resnet.out_channels
                    skip_conv = nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1)
                    skip_conv_list.append(skip_conv)
                if downsample_block.downsamplers is not None:
                    skip_downsampler = nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1)
                    skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "max_pooling":
            skip_first_conv = MaxPoolWithIndices()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                for resnet in downsample_block.resnets:
                    skip_in_channels = resnet.out_channels
                    skip_conv = MaxPoolWithIndices()
                    skip_conv_list.append(skip_conv)
                if downsample_block.downsamplers is not None:
                    skip_downsampler = MaxPoolWithIndices()
                    skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "single_conv_silu":
            skip_first_conv = nn.Sequential(
                nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1),
                nn.SiLU())
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                for resnet in downsample_block.resnets:
                    skip_in_channels = resnet.out_channels
                    skip_conv = nn.Sequential(
                        nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1),
                        nn.SiLU())
                    skip_conv_list.append(skip_conv)
                if downsample_block.downsamplers is not None:
                    skip_downsampler = nn.Sequential(
                        nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1),
                        nn.SiLU())
                    skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        
    
    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], encoder_hidden_states: torch.Tensor, **kwargs):
    
        #sample = kwargs.get('sample')
        #timestep = kwargs.get('timestep')
        #encoder_hidden_states = kwargs.get('encoder_hidden_states')
        class_labels = kwargs.get('class_labels', None)
        timestep_cond = kwargs.get('timestep_cond', None)
        attention_mask = kwargs.get('attention_mask', None)
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs', None)
        added_cond_kwargs = kwargs.get('added_cond_kwargs', None)
        down_block_additional_residuals = kwargs.get('down_block_additional_residuals', None)
        mid_block_additional_residual = kwargs.get('mid_block_additional_residual', None)
        down_intrablock_additional_residuals = kwargs.get('down_intrablock_additional_residuals', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        return_dict = kwargs.get('return_dict', True)
        
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
            
        # 3.5 operate skip connetion between down and up block
        skip_outputs = []
        for i in range(len(down_block_res_samples)):
            model = self.copy_crop_conv_tuple[i].to(sample.device)
            tensor = down_block_res_samples[i]
            output = model(tensor)
            skip_outputs.append(output)
        down_block_res_samples = tuple(skip_outputs)
        

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        
        #for i in range(len(down_block_res_samples)):
        #    print(i)
        #    print(down_block_res_samples[i].shape)
        
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


class CustomPartialConvCopyCropUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, conv_type="skip_unet", oper_block_list=[-1, 0, 1, 2, 3], custom_conv_num=1, *args, **kwargs):
    
        # conv_type can be chosen from ["skip_unet", "single_conv", "max_pooling", "single_conv_silu"]
        # oper_block_list suggest which block to have conv connection, "-1" represent the first connection layer which use the sample before the unet 
        
        super().__init__(*args, **kwargs)
        block_out_channels = kwargs.get('block_out_channels', None)
        if conv_type == "skip_unet":
            if -1 in oper_block_list:
                skip_first_conv = SkipUNet(in_channel=block_out_channels[0])
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = SkipUNet(in_channel=skip_in_channels)
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = SkipUNet(in_channel=block_out_channels[i])
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
                
        elif conv_type == "single_conv":
            if -1 in oper_block_list:
                skip_first_conv = nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1)
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1)
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1)
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "custom_conv":
            if -1 in oper_block_list:
                skip_first_conv = nn.Sequential()
                for i in range(custom_conv_num):
                    skip_first_conv.add_module(f'block_-1_conv_{i+1}', nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1))
                    skip_first_conv.add_module(f'block_-1_silu_{i+1}', nn.SiLU())
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                idx = 0
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = nn.Sequential()
                        for j in range(custom_conv_num):
                            skip_conv.add_module(f'block_{i}_conv_{idx}', nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1))
                            skip_conv.add_module(f'block_{i}_silu_{idx}', nn.SiLU())
                            idx += 1
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Sequential()
                        for j in range(custom_conv_num):
                            skip_downsampler.add_module(f'block_{i}_conv_{idx}', nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1))
                            skip_downsampler.add_module(f'block_{i}_silu_{idx}', nn.SiLU())
                            idx += 1
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "max_pooling":
            if -1 in oper_block_list:
                skip_first_conv = MaxPoolWithIndices()
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = MaxPoolWithIndices()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = MaxPoolWithIndices()
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "single_conv_silu":
            if -1 in oper_block_list:
                skip_first_conv = nn.Sequential(
                    nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1),
                    nn.SiLU())
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = nn.Sequential(
                            nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1),
                            nn.SiLU())
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Sequential(
                            nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1),
                            nn.SiLU())
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        
    
    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], encoder_hidden_states: torch.Tensor, **kwargs):
        #print(sample.shape)
        #sample = kwargs.get('sample')
        #timestep = kwargs.get('timestep')
        #encoder_hidden_states = kwargs.get('encoder_hidden_states')
        class_labels = kwargs.get('class_labels', None)
        timestep_cond = kwargs.get('timestep_cond', None)
        attention_mask = kwargs.get('attention_mask', None)
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs', None)
        added_cond_kwargs = kwargs.get('added_cond_kwargs', None)
        down_block_additional_residuals = kwargs.get('down_block_additional_residuals', None)
        mid_block_additional_residual = kwargs.get('mid_block_additional_residual', None)
        down_intrablock_additional_residuals = kwargs.get('down_intrablock_additional_residuals', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        return_dict = kwargs.get('return_dict', True)
        
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
            
        # 3.5 operate skip connetion between down and up block
        skip_outputs = []
        for i in range(len(down_block_res_samples)):
            model = self.copy_crop_conv_tuple[i].to(sample.device)
            tensor = down_block_res_samples[i]
            output = model(tensor)
            skip_outputs.append(output)
        down_block_res_samples = tuple(skip_outputs)
        

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        
        #for i in range(len(down_block_res_samples)):
        #    print(i)
        #    print(down_block_res_samples[i].shape)
        
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


class CustomPartialConvCopyCropUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, conv_type="skip_unet", oper_block_list=[-1, 0, 1, 2, 3], custom_conv_num=1, *args, **kwargs):
    
        # conv_type can be chosen from ["skip_unet", "single_conv", "max_pooling", "single_conv_silu"]
        # oper_block_list suggest which block to have conv connection, "-1" represent the first connection layer which use the sample before the unet 
        
        super().__init__(*args, **kwargs)
        block_out_channels = kwargs.get('block_out_channels', None)
        if conv_type == "skip_unet":
            if -1 in oper_block_list:
                skip_first_conv = SkipUNet(in_channel=block_out_channels[0])
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = SkipUNet(in_channel=skip_in_channels)
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = SkipUNet(in_channel=block_out_channels[i])
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
                
        elif conv_type == "single_conv":
            if -1 in oper_block_list:
                skip_first_conv = nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1)
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1)
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1)
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "custom_conv":
            if -1 in oper_block_list:
                skip_first_conv = nn.Sequential()
                for i in range(custom_conv_num):
                    skip_first_conv.add_module(f'block_-1_conv_{i+1}', nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1))
                    skip_first_conv.add_module(f'block_-1_silu_{i+1}', nn.SiLU())
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                idx = 0
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = nn.Sequential()
                        for j in range(custom_conv_num):
                            skip_conv.add_module(f'block_{i}_conv_{idx}', nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1))
                            skip_conv.add_module(f'block_{i}_silu_{idx}', nn.SiLU())
                            idx += 1
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Sequential()
                        for j in range(custom_conv_num):
                            skip_downsampler.add_module(f'block_{i}_conv_{idx}', nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1))
                            skip_downsampler.add_module(f'block_{i}_silu_{idx}', nn.SiLU())
                            idx += 1
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "max_pooling":
            if -1 in oper_block_list:
                skip_first_conv = MaxPoolWithIndices()
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = MaxPoolWithIndices()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = MaxPoolWithIndices()
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        elif conv_type == "single_conv_silu":
            if -1 in oper_block_list:
                skip_first_conv = nn.Sequential(
                    nn.Conv2d(in_channels=block_out_channels[0], out_channels=block_out_channels[0], kernel_size=3, padding=1),
                    nn.SiLU())
            else:
                skip_first_conv = nn.Identity()
            self.copy_crop_conv_tuple = (skip_first_conv,)
            
            for i, downsample_block in enumerate(self.down_blocks):
                skip_conv_list = []
                if i in oper_block_list:
                    for resnet in downsample_block.resnets:
                        skip_in_channels = resnet.out_channels
                        skip_conv = nn.Sequential(
                            nn.Conv2d(in_channels=skip_in_channels, out_channels=skip_in_channels, kernel_size=3, padding=1),
                            nn.SiLU())
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Sequential(
                            nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i], kernel_size=3, padding=1),
                            nn.SiLU())
                        skip_conv_list.append(skip_downsampler)
                else:
                    for resnet in downsample_block.resnets:
                        skip_conv = nn.Identity()
                        skip_conv_list.append(skip_conv)
                    if downsample_block.downsamplers is not None:
                        skip_downsampler = nn.Identity()
                        skip_conv_list.append(skip_downsampler)
                skip_conv_list = tuple(skip_conv_list)
                self.copy_crop_conv_tuple += skip_conv_list
        
        
    
    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], encoder_hidden_states: torch.Tensor, **kwargs):
        #print(sample.shape)
        #sample = kwargs.get('sample')
        #timestep = kwargs.get('timestep')
        #encoder_hidden_states = kwargs.get('encoder_hidden_states')
        class_labels = kwargs.get('class_labels', None)
        timestep_cond = kwargs.get('timestep_cond', None)
        attention_mask = kwargs.get('attention_mask', None)
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs', None)
        added_cond_kwargs = kwargs.get('added_cond_kwargs', None)
        down_block_additional_residuals = kwargs.get('down_block_additional_residuals', None)
        mid_block_additional_residual = kwargs.get('mid_block_additional_residual', None)
        down_intrablock_additional_residuals = kwargs.get('down_intrablock_additional_residuals', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        return_dict = kwargs.get('return_dict', True)
        
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
            
        # 3.5 operate skip connetion between down and up block
        skip_outputs = []
        for i in range(len(down_block_res_samples)):
            model = self.copy_crop_conv_tuple[i].to(sample.device)
            tensor = down_block_res_samples[i]
            output = model(tensor)
            skip_outputs.append(output)
        down_block_res_samples = tuple(skip_outputs)
        

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        
        #for i in range(len(down_block_res_samples)):
        #    print(i)
        #    print(down_block_res_samples[i].shape)
        
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

class DualUNet(ModelMixin):
    def __init__(self, unet_front, unet_tail, t_threshold, config=None, *args, **kwargs):
        super().__init__()
        self.unet_front = unet_front
        self.unet_tail = unet_tail
        self.t_threshold = t_threshold
        self.config = config if config is not None else self._create_config()
    
    def _create_config(self):
        # Create a default config, or merge the configs of unet_front and unet_tail
        config = {
            "_class_name": "DualUNet",
            "_diffusers_version": "0.18.2",
            "in_channels": 4,
            "out_channels": 4,
            "attention_head_dim": [5, 10, 20, 20],
            # Add other necessary shared configurations here
        }
        return config

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, **kwargs):
        front_mask = timestep < self.t_threshold
        tail_mask = timestep >= self.t_threshold
        if front_mask.any():
            front_samples = sample[front_mask].squeeze(0)
            front_timesteps = timestep[front_mask]
            front_encoder_hidden_states = encoder_hidden_states[front_mask].squeeze(0)
            front_output = self.unet_front(front_samples, front_timesteps, front_encoder_hidden_states, **kwargs)
        else:
            front_output = None

        if tail_mask.any():
            tail_samples = sample[tail_mask].squeeze(0)
            tail_timesteps = timestep[tail_mask]
            tail_encoder_hidden_states = encoder_hidden_states[tail_mask].squeeze(0)
            tail_output = self.unet_tail(tail_samples, tail_timesteps, tail_encoder_hidden_states, **kwargs)
        else:
            tail_output = None
        
        if front_output is not None:
            return front_output
        else:
            return tail_output














'''
    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], encoder_hidden_states: torch.Tensor, **kwargs):
    
        #sample = kwargs.get('sample')
        #timestep = kwargs.get('timestep')
        #encoder_hidden_states = kwargs.get('encoder_hidden_states')
        class_labels = kwargs.get('class_labels', None)
        timestep_cond = kwargs.get('timestep_cond', None)
        attention_mask = kwargs.get('attention_mask', None)
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs', None)
        added_cond_kwargs = kwargs.get('added_cond_kwargs', None)
        down_block_additional_residuals = kwargs.get('down_block_additional_residuals', None)
        mid_block_additional_residual = kwargs.get('mid_block_additional_residual', None)
        down_intrablock_additional_residuals = kwargs.get('down_intrablock_additional_residuals', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        return_dict = kwargs.get('return_dict', True)
        
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        
        
        #if USE_PEFT_BACKEND:
        #    # weight the lora layers by setting `lora_scale` for each PEFT layer
        #    scale_lora_layers(self, lora_scale)
        

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            
            ############# Change:just set res_hidden_states_tuple=sample, which is the same as hidden_states. ############
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    #res_hidden_states_tuple=res_samples,
                    res_hidden_states_tuple=sample,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    #res_hidden_states_tuple=res_samples,
                    res_hidden_states_tuple=sample,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        #if USE_PEFT_BACKEND:
        #    # remove `lora_scale` from each PEFT layer
        #    unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
        
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
        
'''
