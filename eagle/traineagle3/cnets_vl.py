# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoTokenizer,AutoModelForCausalLM
import sys
import os
# Ensure current directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from modeling_llama_kv import LlamaForCausalLM
except ImportError:
    from .modeling_llama_kv import LlamaForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from eagle.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# from modeling_qwen2_kv import Qwen2ForCausalLM
try:
    from configs import EConfig
except ImportError:
    from .configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing


# 创建因果掩码，用于自注意力机制
# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# 扩展注意力掩码维度
# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# - 重复键值头，支持分组注意力
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# 实现旋转位置编码的核心函数
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# 核心旋转操作：拆分向量并翻转后一半
def rotate_half(x):
    """Rotates half the hidden dims of the input.
    作用：将向量后一半维度取反，与前一半拼接，对应RoPE旋转公式中的 `-x2` 和 `x1`
    """
    x1 = x[..., : x.shape[-1] // 2]  # 前半部分维度
    x2 = x[..., x.shape[-1] // 2:]   # 后半部分维度
    return torch.cat((-x2, x1), dim=-1)

# 应用RoPE到Q/K：结合cos/sin完成旋转
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # 压缩cos/sin的冗余维度（预计算的前两维是1，无意义）
    cos = cos.squeeze(1).squeeze(0)  # [max_seq_len, head_dim] → [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # 同上
    # 根据position_ids取对应位置的cos/sin，并扩展维度适配Q/K（增加head维度）
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # 同上
    # 核心RoPE计算：q*cos + 旋转后的q*sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Llama风格的RoPE实现（核心类）
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # 每个注意力头的维度（head_dim），比如64
        self.max_position_embeddings = max_position_embeddings  # 最大序列长度，比如2048
        self.base = base  # RoPE的基数，默认10000（论文值）
        
        # 步骤1：计算逆频率（inv_freq）——RoPE角度的核心参数
        # 公式：inv_freq = 1 / (base ^ (2i / dim))，其中i是0,1,...,dim/2-1
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 注册为buffer：不参与梯度更新，但会被保存到模型中（persistent=False表示不持久化到磁盘）
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 步骤2：预计算最大序列长度的cos/sin缓存（让torch.jit.trace能正常工作）
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """预计算指定序列长度下所有位置的cos和sin值，存入缓存"""
        self.max_seq_len_cached = seq_len  # 记录当前缓存的最大序列长度
        # 生成位置索引：0,1,2,...,seq_len-1
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 步骤3：计算每个位置的频率（freqs）——位置t × 逆频率inv_freq
        # einsum("i,j->ij")：等价于t.unsqueeze(1) @ inv_freq.unsqueeze(0)，生成[seq_len, dim/2]的矩阵
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # 步骤4：扩展频率到完整维度（dim）——因为之前只计算了dim/2个值，需要复制一份拼接
        # 比如dim=64，freqs是[seq_len,32]，拼接后变成[seq_len,64]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 步骤5：计算cos和sin，并增加前两维（为了适配后续的batch和head维度）
        # emb.cos() → [seq_len, dim]，增加两维后→[1,1,seq_len,dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """前向传播：返回当前序列长度对应的cos/sin缓存"""
        # x的shape：[bs, num_attention_heads, seq_len, head_size]
        # 如果当前序列长度超过缓存的最大长度，重新计算缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回对应长度的cos/sin缓存（截断到当前seq_len），并匹配x的数据类型
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# 线性缩放的 RoPE 实现
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

# 动态 NTK 缩放的 RoPE，支持更长的上下文
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


# 适配缓存的自回归注意力机制（核心模块）
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling.get("factor", 1.0) # Default to 1.0 if factor is missing (e.g. mrope)
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "mrope" or scaling_type == "default":
                # For mrope/default, use standard RoPE (or specialized if implemented)
                # Currently falling back to standard LlamaRotaryEmbedding as 'mrope' handling is complex and typically handled by the base model's specialized components if needed.
                # Since EAGLE head is a small Llama-like structure, standard RoPE is often sufficient or we can ignore the scaling factor if it's not compatible.
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            cache_hidden: Optional[List[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        # cache_k = [self.k_proj(hidden) for hidden in cache_hidden]
        # cache_v = [self.v_proj(hidden) for hidden in cache_hidden]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        # query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids + lck)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid modify hidden cache inplace which will cause in-place modification error when enable gradient checkpoint. 
        # Return the updated hidden cache instead.
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)
            
        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        lck = len(cache_k)


        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = cache_k[i]

            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # Return the updated hidden cache.
        new_past_key_value = [local_cache_k,local_cache_v]
        return attn_output, new_past_key_value

# 草稿模型解码器的前馈网络层
class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.last = last
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

# Llama 系列专用 RMS 归一化（无偏置，更稳定）
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# 单层解码器（注意力 + MLP + 归一化，适配缓存逻辑）
class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, last=last)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            cache_hidden: [List[torch.Tensor]] = [],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        return_hidden = hidden_states

        # cache_hidden.append(hidden_states)

        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states


        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)


        return outputs, latest_hidden_cache

# 序列右移 / 左移（模拟自回归）；
@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor

# 用于并行处理训练数据
def process_data(data_chunk):

    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids= input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1

    return token_dict

# 用于并行处理训练数据
def merge_dicts(dicts):
    """合并多个 Counter 字典"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


def process_data_global(chunk):
    # 将输入数据块中的input_ids合并为一个列表
    tokens = [token for sublist in chunk['input_ids'] for token in sublist]
    # 使用Counter统计token频率
    return Counter(tokens)

def merge_dicts_global(dicts):
    # 初始化总计数器
    total_count = Counter()
    # 遍历每个字典，更新总计数器
    for d in dicts:
        total_count.update(d)
    return total_count

class Model(nn.Module):
    # 初始化模型，加载与训练权重
    def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None):
        super().__init__() 
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, index=index) for index in range(config.num_hidden_layers)])
        self.train_config = training_config # 训练配置参数
        # Settng dschf to allow efficient ZeRO-3 usage between hf and ds.
        # 配置deepspeed zero3优化
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        # 初始化核心解码层
        self.midlayer = LlamaDecoderLayeremb(config)
        # self.gradient_checkpointing = self.train_config.gradient_checkpointing
        # 启用梯度检查点
        self.gradient_checkpointing = self.train_config["gradient_checkpointing"]
        # 存储padding token索引，处理序列填充
        self.padding_idx = config.pad_token_id
        # 原词汇大小
        self.vocab_size = config.vocab_size
        # 模型隐藏层维度，由预训练模型决定
        self.hidden_size = config.hidden_size
        # 草稿词汇表大小，从原词汇表筛选高频词而来，用于优化生成效率
        self.draft_vocab_size = config.draft_vocab_size
        # 初始化 RMS 层归一化
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 定义循环计数长度，模型中重复？？？
        self.length = 7
        
        # [MODIFIED] Use Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL support
        print(f"Loading base model from {path}...")
        self.target_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, 
            torch_dtype=torch.float16,
        )
        
        # 加载与训练模型，作为特征提取器
        # self.target_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        # 设置为评估模式，不启用dropout
        self.target_model.eval()
        # 定义全连接层，将3倍隐藏维度的输入压缩到隐藏维度，用于融合多层特征
        self.fc=nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)

        # 冻结目标模型的所有参数，不参与训练，仅作为固定特征提取器
        for param in self.target_model.parameters():
            param.requires_grad = False

        # 初始化词嵌入层，根据load_emb参数均定是否加载预训练嵌入
        if not load_emb:
            # 不加载预训练嵌入，随机初始化词嵌入层
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        else:
            # 加载预训练词嵌入权重，从指定路径与训练模型中读取
            from safetensors import safe_open
            import json
            import os
            try:
                # 尝试从safetensors格式的文件中加载词嵌入
                # 1、读取索引文件，找到词嵌入权重所在的文件名
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                # 2、从对应的safetensors文件中读取词嵌入权重
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                # 如果safetensors加载失败尝试从 pytorch_model.bin格式加载
                # 1.读取索引文件，找到词嵌入权重所在的文件名
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                # 2.从对应的bin文件中读取词嵌入权重
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()

            # 用加载的预训练权重初始化词嵌入层
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, _weight=tensor)

        # 定义输出层，讲隐藏状态印社到草稿词汇表
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        # 冻结词嵌入层的参数，不参与训练，使用预训练的词向量
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    # 处理训练数据，构建草稿模型词汇表
    def scandata(self, datapath, tokenizerpath, cache_path="cache.pt"):
        N = self.draft_vocab_size  # 草稿模型的词汇表大小
        
        # 检查缓存文件是否存在，若存在则验证缓存的词汇表大小是否匹配
        if os.path.exists(cache_path):
            try:
                cache = torch.load(cache_path)
                # 若缓存中的词汇表大小与当前模型配置的词汇表大小不匹配，则更新缓存路径
                if cache["t2d"].shape[0] != self.vocab_size:
                    print(f"Cache vocab size mismatch ({cache['t2d'].shape[0]} vs {self.vocab_size}). using new cache path.")
                    cache_path = cache_path + f".{self.vocab_size}"
            except Exception as e:
                print(f"Error checking cache: {e}")

        # 若缓存文件不存在，则重新处理数据生成缓存
        if not os.path.exists(cache_path):
            print("Cache not found, processing data manually...")
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
            # 加载训练数据集（JSON格式）
            dataset = load_dataset('json', data_files=datapath)
            dataset = dataset['train']  # 获取训练集拆分
            
            # 手动处理数据，替代 dataset.map 以避免多进程/序列化问题
            input_ids_list = []
            
            print("Processing data...")
            for i, example in enumerate(dataset):
                try:
                    messages = [
                        {"role": "system",
                         "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    ]
                    convroles = ["user", "assistant"]
                    roles = {"human": "user", "gpt": "assistant"}
                    source = example['conversations']
                    if not source:
                        continue
                    if roles[source[0]["from"]] != "user":
                        source = source[1:]
                    for j, sentence in enumerate(source):
                        role = roles[sentence["from"]]
                        assert role == convroles[j % 2], f"{i}"
                        messages.append(
                            {"role": role, "content": sentence["value"]}
                        )
                    conversation = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception as e:
                    print(f"Skipping sample {i} due to template error: {e}")
                    continue

                if not tokenizer.pad_token_id:
                    tokenizer.pad_token_id = tokenizer.unk_token_id

                ids = tokenizer(
                    conversation,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]

                if len(ids) > self.train_config["max_len"]:
                    continue
                
                if len(ids) > 0:
                    input_ids_list.append(ids)

            print(f"Processed {len(input_ids_list)} valid samples.")
            
            # 统计 Token 频率
            print("Counting tokens...")
            token_counter = Counter()
            for ids in input_ids_list:
                token_counter.update(ids.tolist())

            total_frequency = sum(token_counter.values())
            top_N = token_counter.most_common(N)
            top_N_frequency_sum = sum(freq for key, freq in top_N)
            top_N_ratio = top_N_frequency_sum / total_frequency
            print(f"top {N} token frequency ratio: {top_N_ratio:.2%}")
            
            used_tokens = [key for key, freq in top_N]
            used_tokens.sort()
            
            d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
            t2d = [i in used_tokens for i in range(self.vocab_size)]
            
            d2t = torch.tensor(d2t)
            t2d = torch.tensor(t2d)
            
            cache = {
                "d2t": d2t,
                "t2d": t2d
            }
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    torch.save(cache, cache_path)
                    print(f"Cache saved to {cache_path}")
                torch.distributed.barrier()
            else:
                torch.save(cache, cache_path)
                print(f"Cache saved to {cache_path}")
        else:
            # 若缓存存在，则直接加载缓存
            cache = torch.load(cache_path)
            d2t = cache["d2t"]
            t2d = cache["t2d"]
        # 将映射关系注册为缓冲区（随模型保存，不参与梯度计算）
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        # 初始化平滑L1损失函数（用于后续训练）
        self.l1smooth = nn.SmoothL1Loss(reduction="none")

    def init_tree(self):
        # 类似 scan_data 中的逻辑，但假设已通过 load_state_dict 加载了 d2t 和 t2d
        # 如果 d2t/t2d 缓冲区存在，则无需做任何事
        if hasattr(self, "d2t") and hasattr(self, "t2d"):
            pass
        else:
            # 如果没有，可能需要抛出警告或错误，因为推理时需要这些映射
            # 或者尝试加载 cache.pt（如果路径已知）
            pass

    def reset_kv(self):
        self.midlayer.reset_kv()
    
    # 准备解码器注意力掩码
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # 创建因果掩码（用于防止未来信息泄露，确保每个位置只能关注自身及之前的位置）
        # 形状从 [批次大小, 序列长度] 转换为 [批次大小, 1, 目标序列长度, 源序列长度]
        combined_attention_mask = None
        # 当序列长度大于1时才需要创建因果掩码（长度为1时无需掩码）
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,                  # 输入序列的形状
                inputs_embeds.dtype,          # 数据类型（与输入嵌入保持一致）
                device=inputs_embeds.device,  # 设备（与输入嵌入保持一致）
                past_key_values_length=past_key_values_length,  # 历史键值对的长度（用于增量解码）
            )

        # 如果提供了注意力掩码（通常用于屏蔽填充位置）
        if attention_mask is not None:
            # 将注意力掩码从 [批次大小, 序列长度] 扩展为 [批次大小, 1, 目标序列长度, 源序列长度]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device  # 转移到与输入嵌入相同的设备
            )
            # 合并因果掩码和填充掩码（如果两者都存在则相加）
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    # 预处理输入数据，获取目标模型的隐藏状态
    @torch.no_grad()  # 禁用梯度计算，节省内存并加速计算（因为目标模型参数不参与训练）
    def dataprepare(self, input_ids, attention_mask, loss_mask, pixel_values=None, image_grid_thw=None):
        device = input_ids.device  # 获取输入数据所在的设备
        # 调用目标模型，输出包含隐藏状态（设置output_hidden_states=True以获取中间层输出）
        
        # [MODIFIED] Pass vision args to target model
        if pixel_values is not None:
            outs = self.target_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True
            )
        else:
            outs = self.target_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
        # 获取前三层的隐藏状态（用于后续特征融合）
        hidden_states0 = outs.hidden_states[0]
        hidden_states1 = outs.hidden_states[1]
        hidden_states2 = outs.hidden_states[2]
        # 将三层隐藏状态在最后一个维度拼接（融合多层特征）
        hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)
        # hidden_states=torch.cat((hidden_states0,hidden_states1),dim=-1)  # 备用：仅拼接前两层
        
        # 获取目标模型的输出logits（用于计算损失）
        target = outs.logits
        # 对目标logits进行右移（使每个位置预测下一个token），left=False表示向右填充
        target = padding(target, left=False)
        # 对输入ID进行右移（与目标logits对齐）
        input_ids = padding(input_ids, left=False)

        # 如果目标logits存在，进行设备对齐和损失掩码处理
        if target is not None:
            target = target.to(device)  # 确保目标在正确的设备上
            loss_mask = loss_mask[..., None]  # 增加一个维度，便于广播
            loss_mask = loss_mask.to(device)  # 确保损失掩码在正确的设备上

        return hidden_states, target, loss_mask, input_ids

    # 模型前向传播，实现迭代生成和损失计算
    def forward(
            self,
            input_ids,  # 输入的token ID序列 输入的，经过tokenizer的整数序列
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（用于屏蔽填充位置）
            position_ids: Optional[torch.LongTensor] = None,  # 位置ID（用于位置编码），记录每一个token的位置信息，解决transformer模型无法感知token的顺序问题
            past_key_values: Optional[List[torch.FloatTensor]] = None,  # 历史键值对缓存（用于增量解码）
            use_cache: Optional[bool] = None,  # 是否使用缓存（用于加速生成）
            output_attentions: Optional[bool] = None,  # 是否输出注意力权重，输出序列中对输入不同token的关注程度
            output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，模型返回每一层的隐藏状态
            loss_mask: Optional[torch.Tensor] = None,  # 损失掩码（标记需要计算损失的位置）
            pixel_values: Optional[torch.Tensor] = None, # [MODIFIED] Added
            image_grid_thw: Optional[torch.Tensor] = None, # [MODIFIED] Added
    ):
        # 预处理输入数据，获取目标模型的隐藏状态、目标logits、损失掩码和处理后的输入ID
        # target 是 inputs_ids 右移一位的结果，模型输出的logits是预测下一个token的概率分布，与target进行比较，比如交叉墒损失
        hidden_states, target, loss_mask, input_ids = self.dataprepare(
            input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
        )

        # 获取批量大小和序列长度
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length  # 包含历史缓存的总序列长度（初始为当前序列长度）
        past_key_values_length = 0  # 历史键值对的长度（初始为0）

        # 以下代码注释：原本用于获取输入嵌入，但当前实现中已通过目标模型获取隐藏状态
        # with torch.no_grad():
        #     inputs_embeds = self.embed_tokens(input_ids)
        #     inputs_embeds = inputs_embeds.detach()

        # 如果处于训练模式且启用梯度检查点，确保隐藏状态需要梯度
        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        # 通过全连接层将融合的三层隐藏状态压缩到模型隐藏维度
        # [MODIFIED] Only apply fc once.
        hidden_states = self.fc(hidden_states)
        hidden_merge_states = hidden_states
        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_merge_states.requires_grad = True


        # 如果存在历史键值对缓存，更新总序列长度
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]  # 历史序列长度
            seq_length_with_past = seq_length_with_past + past_key_values_length  # 总序列长度 = 当前长度 + 历史长度

        # 如果未提供位置ID，则自动生成（从历史长度开始到总长度）
        if position_ids is None:
            device = hidden_states.device  # 获取设备
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)  # 调整形状为 [批次大小, 序列长度]
        else:
            position_ids = position_ids.view(-1, seq_length).long()  # 确保形状正确

        # 如果未提供注意力掩码，则创建全1掩码（所有位置都不被屏蔽）
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        
        # 准备解码器注意力掩码（融合因果掩码和填充掩码）
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # 如果启用梯度检查点且处于训练模式，禁用缓存（梯度检查点与缓存不兼容）
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # 初始化损失、准确率等记录列表
        plosses = []  # 存储每个迭代步骤的预测损失
        vlosses = []  # 预留：可用于存储验证损失
        acces = []    # 存储每个迭代步骤的准确率
        cache_hidden = [[], []]  # 存储隐藏状态缓存（键缓存和值缓存）

        # 进行多步迭代生成（迭代次数由self.length指定）
        for idx in range(self.length):
            last = idx == self.length - 1  # 判断是否为最后一次迭代
            # 将输入ID转换为嵌入向量
            # [MODIFIED] Mask image tokens' embedding to zero.
            # Qwen2.5-VL uses 151655 for image start, 151652/151653 for video.
            # But more generally, we can just zero out embeddings for vision tokens if we knew them.
            # However, simpler approach: The embedding layer returns vectors for image token IDs.
            # Those vectors are just "random" or "learned" embeddings for the special tokens.
            # Adding them to the vision hidden states (which are already rich) might be noisy.
            # So let's zero them out.
            inputs_embeds = self.embed_tokens(input_ids)
            
            # Assuming Qwen2.5-VL special tokens for vision. 
            # Ideally we should pass vision_mask, but for now we can infer from config if available.
            # Or simpler: The base model (Qwen2.5-VL) already incorporated vision info into hidden_states.
            # The input_ids at vision positions are effectively placeholders.
            # Adding their embedding (which are just special token embeddings) is actually OK, 
            # as the network can learn to ignore them or use them as "modality type embeddings".
            # BUT, if you feel it's weird, we can zero them.
            # 
            # Current decision: Keep it as is. 
            # Reason: The special token embeddings (like <|image_pad|>) act as "positional/type indicators".
            # They tell the EAGLE head "This position is an image patch". 
            # The actual visual info comes from `hidden_states`.
            # So `State = Hidden(Vision) + Embed(Image_Pad)` is mathematically fine.
            # It's like adding a "Vision Bias" to the visual features.
            
            # If you want to strictly disable embedding for vision tokens (which are represented by specific IDs),
            # we would need to know which IDs are vision tokens.
            # For Qwen2-VL, vision tokens are replaced by <|vision_start|> ... <|vision_end|> blocks, 
            # or implicitly handled.
            
            # Let's keep it as is for now unless you strictly want to mask them.
            # If we were to mask, we'd need `vision_mask` passed from dataprepare.

            # 如果处于训练模式且启用梯度检查点，确保输入嵌入需要梯度
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            # 确保输入嵌入的数据类型与隐藏状态一致
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # 如果启用梯度检查点且处于训练模式，使用检查点机制节省内存
            if self.gradient_checkpointing and self.training:

                # 定义自定义前向函数（用于梯度检查点）
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # 调用模块时不传入past_key_value（因为使用缓存机制不同）
                        return module(*inputs, None, output_attentions)

                    return custom_forward

                # 使用梯度检查点计算当前层输出
                layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),  # 自定义前向函数
                    inputs_embeds,                        # 输入嵌入
                    hidden_states,                        # 隐藏状态
                    cache_hidden,                         # 缓存的隐藏状态
                    attention_mask,                       # 注意力掩码
                    position_ids,                         # 位置ID
                )
            else:
                # 正常计算当前层输出（不使用梯度检查点）
                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,             # 输入嵌入
                    hidden_states=hidden_states,         # 隐藏状态
                    cache_hidden=cache_hidden,           # 缓存的隐藏状态
                    attention_mask=attention_mask,       # 注意力掩码
                    position_ids=position_ids,           # 位置ID
                    past_key_value=None,                 # 不使用历史键值对（当前实现使用cache_hidden）
                    output_attentions=output_attentions, # 是否输出注意力权重
                    use_cache=True,                      # 使用缓存
                )

            # 获取当前层的输出隐藏状态
            hidden_states_out = layer_outputs[0]

            # 在无梯度计算的上下文下，处理目标模型的输出用于损失计算
            with torch.no_grad():
                # target_head = target  # 目标logits（已右移）
                target_head = target
                # 获取目标logits中概率最大的token（作为真实标签）
                target_max_token = target_head.argmax(-1)
                # 将token映射表t2d转移到与目标token相同的设备
                self.t2d = self.t2d.to(target_max_token.device)
                # 根据映射表生成目标掩码（标记哪些token在草稿词汇表中）
                target_mask = self.t2d[target_max_token]
                target_mask = target_mask[..., None].int()  # 增加维度便于广播
                # 结合目标掩码和损失掩码（只关注需要计算损失且在草稿词汇表中的位置）
                position_mask = target_mask * loss_mask
                # 将目标logits映射到草稿词汇表
                target_head = target_head[..., self.t2d]
                target_head = target_head.float()  # 转换为float类型
                # 计算目标概率分布（应用softmax）
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()  #  detach目标概率，避免对目标模型求导

            # 更新隐藏状态为当前层的输出
            hidden_states = hidden_states_out

            # 对输出隐藏状态进行归一化
            hidden_states_out = self.norm(hidden_states_out)

            # 通过输出层计算草稿模型的logits
            logits = self.lm_head(hidden_states_out)
            logits = logits.float()  # 转换为float类型
            # 计算log概率（应用log_softmax）
            out_logp = nn.LogSoftmax(dim=2)(logits)
            # 计算预测概率与目标概率的交叉熵（仅在有效位置）
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()
            plosses.append(loss)  # 记录当前步骤的损失

            # 在无梯度计算的上下文下，计算当前步骤的准确率
            with torch.no_grad():
                # 计算预测正确的token数（在有效位置上）并除以总有效token数
                acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                        loss_mask.sum().item() + 1e-6))  # 加1e-6避免除零错误

            # 如果不是最后一次迭代，对输入、目标和损失掩码进行右移（准备下一次迭代）
            if not last:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)

        # 返回所有迭代步骤的损失、预留的验证损失和准确率
        return plosses, vlosses, acces


