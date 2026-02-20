# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core xdimo Model                                #
#################################################################################
class MoETransformerBlock(nn.Module):
    """
    A xdimo transformer block with MoE (Mixture of Experts) layer.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, num_experts=4, top_k=2, 
                 expert_capacity_factor=1.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # MoE components
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        
        # Experts
        mlp_hidden_dim = int(hidden_size * mlp_ratio)   
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.experts = nn.ModuleList([
            Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, 
                act_layer=approx_gelu, drop=0)
            for _ in range(num_experts)
        ])
        
        # Gate network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Adaptive modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Attention branch
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MoE branch
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        moe_out = self.moe_forward(x_norm)
        x = x + gate_mlp.unsqueeze(1) * moe_out
        
        return x

    def moe_forward(self, x):
        """
        Mixture of Experts forward pass.
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten sequence dimension
        x_flat = x.reshape(-1, hidden_dim)
        num_tokens = x_flat.size(0)
        
        # Calculate expert capacity
        expert_capacity = min(
            num_tokens, 
            int(self.expert_capacity_factor * num_tokens / self.num_experts)
        )
        
        # Gate logits
        gate_logits = self.gate(x_flat)
        
        # Top-k routing
        topk_logits, topk_indices = torch.topk(
            gate_logits, k=self.top_k, dim=-1
        )
        topk_gates = torch.softmax(topk_logits, dim=-1)
        
        # Create mask for routing
        mask = torch.zeros(
            self.num_experts, num_tokens, 
            dtype=torch.bool, device=x.device
        )
        
        # Assign tokens to experts
        expert_assignments = []
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            mask.index_put_((expert_idx, torch.arange(num_tokens, device=x.device)), 
                           torch.tensor(True))
            expert_assignments.append(expert_idx)
        
        # Process through experts
        expert_outputs = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            # Get tokens assigned to this expert
            expert_mask = mask[expert_idx]
            tokens = x_flat[expert_mask]
            
            if tokens.size(0) > 0:
                # Process tokens through expert
                expert_out = self.experts[expert_idx](tokens)
                
                # Apply gating weights
                for k in range(self.top_k):
                    k_mask = (expert_assignments[k][expert_mask] == expert_idx)
                    k_gates = topk_gates[expert_mask, k][k_mask]
                    expert_out[k_mask] *= k_gates.unsqueeze(1)
                
                expert_outputs[expert_mask] += expert_out
        
        return expert_outputs.reshape(batch_size, seq_len, hidden_dim)


class TransformerBlock(nn.Module):
    """
    A xdimo tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of xdimo.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class xdimo(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=1,
        attention_mode='math',
        num_experts=4,        # Number of experts in MoE
        top_k=2,              # Top-k experts to route to
        expert_capacity_factor=1.0,  # Capacity factor for experts
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        # self.blocks = nn.ModuleList([
        #     TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        # ])
        # MoE parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        
        # Create blocks with MoE
        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                num_experts=num_experts, top_k=top_k, 
                expert_capacity_factor=expert_capacity_factor,
                attention_mode=attention_mode
            ) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in xdimo blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self,
                x,
                t,
                y=None,
                text_embedding=None,
                use_fp16=False,
                verbose=False):
        """
        Forward pass of xdimo.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        verbose: 若为 True，打印每一步的输入/输出张量尺寸及对应公式与数值
        """
        def _log(step, name, inp_shape, out_shape, formula_str=""):
            if not verbose:
                return
            fmt = "  {:<6} {}  |  输入: {}  ->  输出: {}"
            print(fmt.format(step, name, inp_shape, out_shape))
            if formula_str:
                print("           公式: " + formula_str)

        if use_fp16:
            x = x.to(dtype=torch.float16)

        batches, frames, channels, high, weight = x.shape
        # 步骤 1: 输入与重排
        _log("1", "输入视频 x", str(tuple(x.shape)), str(tuple(x.shape)),
             "x: (B, F, C, H, W) = ({}, {}, {}, {}, {})".format(batches, frames, channels, high, weight))

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        _log("2", "rearrange → 按帧展平", "(B,F,C,H,W)", "(B*F, C, H, W)",
             "B*F = {}*{} = {}".format(batches, frames, batches * frames))

        # 步骤 3: Patch Embedding
        # num_patches = (H/patch_size)*(W/patch_size)
        grid = high // self.patch_size
        num_patches = self.x_embedder.num_patches
        x = self.x_embedder(x) + self.pos_embed
        _log("3", "x_embedder + pos_embed", "(B*F, C, H, W)", "(B*F, num_patches, D)",
             "num_patches = (H/p)^2 = ({}/{})^2 = {}, D = hidden_size = {}".format(
                 high, self.patch_size, num_patches, self.hidden_size))
        _log("", "  → 输出", "", str(tuple(x.shape)), "")

        t = self.t_embedder(t, use_fp16=use_fp16)
        _log("4", "t_embedder(t)", "(B,) 标量步", "(B, D)",
             "t_embedding_dim = {}".format(self.hidden_size))

        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        _log("5", "timestep 复制到 spatial/temp", "(B, D)", "(B*F 或 B*T, D)",
             "spatial: (B, D) -> (B, F, D) 展平 = (B*F, D) = ({}, {}), temp: (B*num_patches, D)".format(
                 batches * frames, self.hidden_size))

        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
            if verbose:
                print("  {:<6} y_embedder + repeat  |  y: (B,) -> (B, D), y_spatial: (B*F, D), y_temp: (B*num_patches, D)".format("5b"))
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            text_embedding_temp = repeat(text_embedding, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        block_step = 6
        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            else:
                c = timestep_spatial

            if verbose:
                print("  ------ Block pair {}/{} (spatial -> temporal) ------".format(i // 2 + 1, len(self.blocks) // 2))
            x_in = x
            x = spatial_block(x, c)
            _log(str(block_step), "Spatial MoE Block", str(tuple(x_in.shape)), str(tuple(x.shape)),
                 "seq = B*F = {}, dim = {}".format(x.shape[0], x.shape[2]))
            block_step += 1

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            _log(str(block_step), "rearrange → 时间维", "(B*F, num_patches, D)", "(B*num_patches, F, D)",
                 "(B*F, T, D) -> (B*T, F, D), T=num_patches={}".format(num_patches))
            block_step += 1

            if i == 0:
                x = x + self.temp_embed
                if verbose:
                    print("           公式: temp_embed: (1, F, D) = (1, {}, {}) broadcast 加".format(self.num_frames, self.hidden_size))

            if self.extras == 2:
                c = timestep_temp + y_temp
            elif self.extras == 78:
                c = timestep_temp + text_embedding_temp
            else:
                c = timestep_temp

            x_in = x
            x = temp_block(x, c)
            _log(str(block_step), "Temporal MoE Block", str(tuple(x_in.shape)), str(tuple(x.shape)),
                 "seq = B*num_patches = {}, F = {}".format(x.shape[0] // batches, x.shape[1]))
            block_step += 1

            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)
            _log(str(block_step), "rearrange → 空间维", "(B*num_patches, F, D)", "(B*F, num_patches, D)",
                 "恢复为 (B*F, T, D) 进入下一对 block")
            block_step += 1

        if self.extras == 2:
            c = timestep_spatial + y_spatial
        else:
            c = timestep_spatial
        x_in = x
        x = self.final_layer(x, c)
        _log(str(block_step), "final_layer", str(tuple(x_in.shape)), "(B*F, num_patches, p^2*out_ch)",
             "linear: D -> patch_size^2*out_channels = {} -> {}*{}*{}".format(
                 self.hidden_size, self.patch_size, self.patch_size, self.out_channels))
        block_step += 1

        x = self.unpatchify(x)
        h = w = int(x.shape[1] ** 0.5)
        _log(str(block_step), "unpatchify", "(B*F, T, p^2*C)", "(B*F, C, H, W)",
             "h=w=sqrt(T)={}, H=W=h*p={}*{}={}".format(h, h, self.patch_size, h * self.patch_size))
        block_step += 1

        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        _log(str(block_step), "rearrange → 输出视频", "(B*F, C, H, W)", "(B, F, C, H, W)",
             "out: (B, F, out_channels, H, W) = ({}, {}, {}, {}, {})".format(
                 batches, frames, self.out_channels, x.shape[3], x.shape[4]))
        if verbose:
            print_forward_flops(self, batches, frames, high, weight)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False, text_embedding=None):
        """
        Forward pass of xdimo, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y=y, use_fp16=use_fp16, text_embedding=text_embedding)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


def get_xdimo_config(model, input_size=32):
    """
    从任意 xdimo 模型提取用于参数量/计算量统计的配置（通用，支持 MoE 与 非 MoE）。
    """
    if not hasattr(model, 'hidden_size'):
        return None
    D = model.hidden_size
    p = model.patch_size if isinstance(model.patch_size, int) else model.patch_size[0]
    num_patches = (input_size // p) ** 2
    cfg = {
        'D': D,
        'p': p,
        'in_ch': model.in_channels,
        'out_ch': model.out_channels,
        'num_patches': num_patches,
        'num_frames': model.num_frames,
        'depth': len(model.blocks),
        'num_heads': model.num_heads,
    }
    block0 = model.blocks[0]
    if hasattr(block0, 'experts'):
        cfg['mlp_hidden'] = block0.experts[0].fc1.out_features
        cfg['num_experts'] = block0.num_experts
        cfg['top_k'] = block0.top_k
        cfg['is_moe'] = True
    else:
        cfg['mlp_hidden'] = block0.mlp.fc1.out_features
        cfg['num_experts'] = 0
        cfg['top_k'] = 0
        cfg['is_moe'] = False
    return cfg


# ---------- 计算量(FLOPs) 公式与数值，通用用于任意 xdimo 变体 ----------
# 约定: FLOPs = 一次乘法+加法计为 2 FLOPs（即 2*MACs）

def _flops_linear(B, in_f, out_f):
    """Linear: 2 * B * in_f * out_f"""
    return 2 * B * in_f * out_f


def _flops_conv2d(B, C_in, C_out, H_out, W_out, k):
    """Conv2d: 2 * B * C_out * H_out * W_out * (C_in * k * k)"""
    return 2 * B * C_out * H_out * W_out * C_in * k * k


def _flops_attention(B, N, D, num_heads):
    """Attention: qkv 6*B*N*D^2, QK^T 2*B*N^2*D, attn@V 2*B*N^2*D, proj 2*B*N*D^2 -> 8*B*N*D^2 + 4*B*N^2*D"""
    return 8 * B * N * D * D + 4 * B * N * N * D


def _flops_layernorm(B, N, D):
    """LayerNorm (无 affine): 约 5*B*N*D (mean, var, normalize)"""
    return 5 * B * N * D


def _flops_adaln_modulation(B, D, num_chunks=6):
    """adaLN: Linear(D -> num_chunks*D), 2*B*D*(num_chunks*D)"""
    return 2 * B * D * (num_chunks * D)


def _flops_mlp(B, N, D, mlp_hidden):
    """MLP: fc1 2*B*N*D*mlp_hidden, fc2 2*B*N*mlp_hidden*D"""
    return 2 * B * N * D * mlp_hidden + 2 * B * N * mlp_hidden * D


def _flops_moe(B, N, D, mlp_hidden, num_experts, top_k):
    """MoE: gate 2*B*N*D*E; 每个 token 走 top_k 个 expert，每个 expert 2*D*mlp_h + 2*mlp_h*D -> 4*B*N*top_k*D*mlp_hidden"""
    gate_flops = 2 * B * N * D * num_experts
    experts_flops = 4 * B * N * top_k * D * mlp_hidden
    return gate_flops + experts_flops


def flops_one_transformer_block(is_moe, B, N, D, num_heads, mlp_hidden, num_experts, top_k):
    """
    单层 Transformer block 的 FLOPs（含 2 个 LayerNorm、Attention、adaLN、MLP 或 MoE）。
    通用: 可用于 spatial（N=num_patches）或 temporal（N=num_frames）。
    """
    ln = 2 * _flops_layernorm(B, N, D)
    attn = _flops_attention(B, N, D, num_heads)
    adaln = _flops_adaln_modulation(B, D, 6)
    if is_moe:
        mlp_part = _flops_moe(B, N, D, mlp_hidden, num_experts, top_k)
    else:
        mlp_part = _flops_mlp(B, N, D, mlp_hidden)
    return ln + attn + adaln + mlp_part


def compute_forward_flops_per_step(model, B, F, H, W):
    """
    按前向传播步骤计算各步 FLOPs，返回 [(step_name, formula_str, flops_value), ...]。
    通用：适用于任意 xdimo 配置（L/2-MoE、XL/2、S/4 等）。
    """
    cfg = get_xdimo_config(model, H)
    if cfg is None:
        return []
    D = cfg['D']
    p = cfg['p']
    in_ch = cfg['in_ch']
    out_ch = cfg['out_ch']
    num_patches = cfg['num_patches']
    num_frames = cfg['num_frames']
    depth = cfg['depth']
    num_heads = cfg['num_heads']
    mlp_hidden = cfg['mlp_hidden']
    num_experts = cfg['num_experts']
    top_k = cfg['top_k']
    is_moe = cfg['is_moe']

    BF = B * F
    BT = B * num_patches
    steps = []

    # 1. 输入 / 2. rearrange：无计算
    steps.append(("1-2", "输入 + rearrange", "0", 0))

    # 3. x_embedder (Conv2d) + pos_embed (加法忽略)
    # Conv: B*F, (H/p)*(W/p), in_ch*p^2 -> D
    H_out = H // p
    W_out = W // p
    f_embed = _flops_conv2d(BF, in_ch, D, H_out, W_out, p)
    steps.append(("3", "x_embedder (Conv2d)", "2*B*F*(H/p)*(W/p)*in_ch*p^2*D = 2*{}*{}*{}*{}*{}".format(BF, H_out, W_out, in_ch, D), f_embed))

    # 4. t_embedder: B, 256->D, D->D
    f_t = 2 * B * 256 * D + 2 * B * D * D
    steps.append(("4", "t_embedder", "2*B*256*D + 2*B*D^2 = 2*{}*256*{} + 2*{}*{}^2".format(B, D, B, D), f_t))

    # 5. timestep repeat / 5b y_embedder：查表与复制，忽略或计 B*D
    steps.append(("5-5b", "timestep/y repeat", "0 (查表与复制)", 0))

    # 6 起: 成对 spatial + temporal block
    block_step = 6
    f_spatial_one = flops_one_transformer_block(is_moe, BF, num_patches, D, num_heads, mlp_hidden, num_experts, top_k)
    f_temporal_one = flops_one_transformer_block(is_moe, BT, num_frames, D, num_heads, mlp_hidden, num_experts, top_k)
    formula_s = "LayerNorm*2+Attn+adaLN+{} (B*F={}, N=num_patches={})".format("MoE" if is_moe else "MLP", BF, num_patches)
    formula_t = "LayerNorm*2+Attn+adaLN+{} (B*T={}, N=F={})".format("MoE" if is_moe else "MLP", BT, num_frames)

    for i in range(0, depth, 2):
        steps.append((str(block_step), "Spatial Block", formula_s, f_spatial_one))
        block_step += 1
        steps.append((str(block_step), "rearrange (时间维)", "0", 0))
        block_step += 1
        if i == 0:
            steps.append(("", "temp_embed 加", "0", 0))
        steps.append((str(block_step), "Temporal Block", formula_t, f_temporal_one))
        block_step += 1
        steps.append((str(block_step), "rearrange (空间维)", "0", 0))
        block_step += 1

    # final_layer: adaLN 2*D*D (c 为 B*F), linear 2*B*F*num_patches*D*(p^2*out_ch)
    f_adaln_final = 2 * BF * D * (2 * D)
    f_linear_final = 2 * BF * num_patches * D * (p * p * out_ch)
    steps.append((str(block_step), "final_layer", "adaLN: 2*B*F*D*2*D; linear: 2*B*F*T*D*(p^2*out_ch)", f_adaln_final + f_linear_final))
    block_step += 1

    # unpatchify / 最后 rearrange：无乘加
    steps.append((str(block_step), "unpatchify + rearrange", "0", 0))

    return steps


def print_forward_flops(model, B, F, H, W):
    """打印前向传播各步计算量（公式 + 代入数值），通用任意 xdimo 模型。"""
    steps = compute_forward_flops_per_step(model, B, F, H, W)
    if not steps:
        print("无法解析模型配置，跳过 FLOPs 打印")
        return
    total = 0
    print("=" * 70)
    print("前向传播计算量 (FLOPs)：公式 + 代入数值")
    print("=" * 70)
    cfg = get_xdimo_config(model, H)
    print("符号: B={}, F={}, H=W={}, D={}, num_patches={}, depth={}, is_moe={}".format(
          B, F, H, cfg['D'], cfg['num_patches'], cfg['depth'], cfg['is_moe']))
    print("-" * 70)
    for step_id, name, formula, flops in steps:
        total += flops
        if step_id:
            print("  步骤 {}  {}  |  FLOPs = {}".format(step_id, name, flops))
        else:
            print("          {}  |  FLOPs = {}".format(name, flops))
        if formula and formula != "0":
            print("           公式: {}".format(formula))
    print("-" * 70)
    print("总 FLOPs = {} = {:.2f}G".format(total, total / 1e9))
    print("=" * 70)


def print_param_count_with_formula(model, input_size=32, num_classes=1000):
    """
    打印 xdimo 模型参数量，包含公式与代入后的具体数值。
    通用：适用于任意 xdimo 变体（MoE 与 非 MoE）。
    """
    cfg = get_xdimo_config(model, input_size)
    if cfg is None:
        total = sum(p.numel() for p in model.parameters())
        print("总参数量: {} = {:.2f}M".format(total, total / 1e6))
        return

    D = cfg['D']
    p = cfg['p']
    in_ch = cfg['in_ch']
    out_ch = model.out_channels
    num_patches = cfg['num_patches']
    depth = cfg['depth']
    num_experts = cfg['num_experts']

    print("=" * 60)
    print("xdimo 模型参数量统计（公式 + 代入数值）")
    print("=" * 60)
    print("符号: D=hidden_size={}, p=patch_size={}, in_ch={}, out_ch={}".format(D, p, in_ch, out_ch))
    print("      num_patches=(H/p)^2={}^2={}, depth={}, num_experts={}".format(
          input_size // p, num_patches, depth, num_experts))
    print("-" * 60)

    total_calc = 0
    mlp_hidden = cfg['mlp_hidden']

    # 1. x_embedder (PatchEmbed: Conv2d)
    w = model.x_embedder.proj.weight
    b = model.x_embedder.proj.bias
    embed_params = w.numel() + (b.numel() if b is not None else 0)
    formula = "in_ch * p^2 * D + D = {}*{}*{} + {} = {}".format(in_ch, p * p, D, D, embed_params)
    print("1. x_embedder (PatchEmbed):  {}  |  {}".format(embed_params, formula))
    total_calc += embed_params

    # 2. t_embedder
    freq_dim = 256
    t0 = freq_dim * D + D
    t1 = D * D + D
    t_params = t0 + t1
    formula = "256*D+D + D*D+D = {} + {} = {}".format(t0, t1, t_params)
    print("2. t_embedder:                {}  |  {}".format(t_params, formula))
    total_calc += t_params

    # 3. y_embedder (若存在)
    if model.extras == 2 and hasattr(model, 'y_embedder'):
        y_params = model.y_embedder.embedding_table.weight.numel()
        formula = "(num_classes+1)*D = {}*{} = {}".format(num_classes + 1, D, y_params)
        print("3. y_embedder:                {}  |  {}".format(y_params, formula))
        total_calc += y_params
    else:
        print("3. y_embedder:                (未使用 extras=2)")

    # 4. pos_embed / temp_embed
    print("4. pos_embed / temp_embed:    (固定，不训练)")

    # 5. Transformer blocks
    block = model.blocks[0]
    one_block = sum(p.numel() for p in block.parameters())
    if cfg['is_moe']:
        attn_params = (3 * D * D + 3 * D) + (D * D + D)
        one_mlp = D * mlp_hidden + mlp_hidden + mlp_hidden * D + D
        moe_mlp_params = num_experts * one_mlp
        gate_params = D * num_experts
        adaln_params = D * (6 * D) + (6 * D)
        formula = "attn(3D^2+3D+D^2+D) + num_experts*(2*D*mlp_h+mlp_h+D) + D*E + adaLN(6D^2+6D)"
        print("5. 单 MoETransformerBlock:   {}  |  {}".format(one_block, formula))
        print("   其中: attn={}, experts={}*{}={}, gate={}, adaLN={}".format(
              attn_params, num_experts, one_mlp, moe_mlp_params, gate_params, adaln_params))
    else:
        print("5. 单 TransformerBlock:      {}  |  (depth={})".format(one_block, depth))
    total_calc += one_block * depth
    print("   depth = {}, blocks 合计:   {}  |  {} * {} = {}".format(
          depth, one_block * depth, one_block, depth, one_block * depth))

    # 6. final_layer
    # norm_final: elementwise_affine=False -> 0; linear: D * (p^2*out_ch) + p^2*out_ch; adaLN: D*2D+2D
    final_linear = model.final_layer.linear
    final_adaln = model.final_layer.adaLN_modulation[-1]
    fl_linear = final_linear.weight.numel() + (final_linear.bias.numel() if final_linear.bias is not None else 0)
    fl_adaln = final_adaln.weight.numel() + final_adaln.bias.numel()
    final_params = fl_linear + fl_adaln
    formula = "D*(p^2*out_ch)+p^2*out_ch + D*2D+2D = {} + {} = {}".format(fl_linear, fl_adaln, final_params)
    print("6. final_layer:               {}  |  {}".format(final_params, formula))
    total_calc += final_params

    # 总计
    actual_total = sum(p.numel() for p in model.parameters())
    print("-" * 60)
    print("公式合计: {}  |  实际 sum(parameters): {}".format(total_calc, actual_total))
    print("总参数量: {} = {:.2f}M".format(actual_total, actual_total / 1e6))
    print("=" * 60)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   xdimo Configs                                  #
#################################################################################
def xdimo_XL_2_MoE(**kwargs):
    num_experts = kwargs.pop('num_experts', 4)
    top_k = kwargs.pop('top_k', 2)
    return xdimo(
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )

def xdimo_L_2_MoE(**kwargs):
    # 从 kwargs 取出 MoE 参数，避免 get_models 已注入时重复传参导致 TypeError
    num_experts = kwargs.pop('num_experts', 4)
    top_k = kwargs.pop('top_k', 2)
    return xdimo(
        depth=24,
        hidden_size=1024,
        patch_size=2,
        num_heads=16,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )

def xdimo_L_4_MoE(**kwargs):
    num_experts = kwargs.pop('num_experts', 4)
    top_k = kwargs.pop('top_k', 2)
    return xdimo(
        depth=24,
        hidden_size=1024,
        patch_size=4,
        num_heads=16,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )


# Similarly for other configurations...
def xdimo_XL_2(**kwargs):
    return xdimo(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def xdimo_XL_4(**kwargs):
    return xdimo(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def xdimo_XL_8(**kwargs):
    return xdimo(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def xdimo_L_2(**kwargs):
    num_experts = kwargs.pop('num_experts', 4)
    top_k = kwargs.pop('top_k', 2)
    return xdimo(
        depth=24,
        hidden_size=1024,
        patch_size=2,
        num_heads=16,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )

def xdimo_L_4(**kwargs):
    return xdimo(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def xdimo_L_8(**kwargs):
    return xdimo(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def xdimo_S_2(**kwargs):
    return xdimo(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def xdimo_S_4(**kwargs):
    return xdimo(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def xdimo_S_8(**kwargs):
    return xdimo(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


xdimo_models = {
    'xdimo-XL/2': xdimo_XL_2,  'xdimo-XL/4': xdimo_XL_4,  'xdimo-XL/8': xdimo_XL_8,
    'xdimo-L/2':  xdimo_L_2,   'xdimo-L/4':  xdimo_L_4,   'xdimo-L/8':  xdimo_L_8,
    'xdimo-S/2':  xdimo_S_2,   'xdimo-S/4':  xdimo_S_4,   'xdimo-S/8':  xdimo_S_8,
    'xdimo-XL/2-MoE':  xdimo_XL_2_MoE, 
    'xdimo-L/2-MoE': xdimo_L_2_MoE,  # 新增 MoE 版本的 L 模型
}


if __name__ == '__main__':
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("测试模型: xdimo-L/2-MoE")
    print("=" * 60)
    print("Device: {}".format(device))

    # 使用 xdimo_models 构建 xdimo-L/2-MoE（extras=2 用于类别条件）
    model = xdimo_models['xdimo-L/2-MoE'](extras=2, num_classes=1000).to(device)
    input_size = 32
    num_frames = 16
    B, F, C, H, W = 2, num_frames, 4, input_size, input_size

    # ---------- 1. 模型参数量（公式 + 具体数值） ----------
    print("\n")
    print_param_count_with_formula(model, input_size=input_size, num_classes=1000)

    # ---------- 2. 前向传播每一步的输入/输出张量尺寸与公式 ----------
    print("\n")
    print("前向传播步骤与张量尺寸（verbose=True）")
    print("-" * 60)
    x = torch.randn(B, F, C, H, W).to(device)
    t = torch.tensor([0]).to(device) if B == 1 else torch.tensor([0, 1]).to(device)
    y = torch.tensor([0]).to(device) if B == 1 else torch.tensor([0, 1]).to(device)

    with torch.no_grad():
        out = model(x, t, y=y, verbose=True)

    print("-" * 60)
    print("最终输出 out.shape = {}".format(tuple(out.shape)))
    print("公式: (B, F, out_channels, H, W) = ({}, {}, {}, {}, {})".format(
          B, F, model.out_channels, H, W))

    # ---------- 3. 梯度检查 ----------
    print("\n梯度检查:")
    x_g = torch.randn(1, num_frames, 4, input_size, input_size).to(device).requires_grad_(True)
    model.train()
    o = model(x_g, torch.tensor([0]).to(device), y=torch.tensor([0]).to(device))
    o.sum().backward()
    print("  input.grad.norm = {:.4f} (应为非零)".format(x_g.grad.norm().item()))

    print("\n全部测试完成 (xdimo-L/2-MoE)。")