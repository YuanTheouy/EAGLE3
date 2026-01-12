import copy
import json
import time
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *

from .kv_cache import initialize_past_key_values
from .cnets import Model
from .configs import EConfig

# EaModel 是一个继承自 nn.Module 的 PyTorch 模型封装类，
# 核心目标是：在原有大模型（如 Llama/Qwen/Mixtral）基础上，集成 Eagle 系列加速策略，实现更快的文本生成。
class EaModel(nn.Module):
    def __init__(
            self,
            base_model,
            base_model_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        # target model 核心属性
        self.base_model = base_model
        self.config = base_model.config
        # 大模型最后一层输出 [B,L,hidden_size] 通过 lm_head 映射到 [B,L,vocab_size]
        # 公式 logits = hidden_states @ lm_head.weight.T + lm_head.bias，转置T是因为 lm_head.weight 是 [vocab_size, hidden_size]
        self.hidden_size = base_model.lm_head.weight.shape[-1] # hidden_size
        self.vocab_size = base_model.lm_head.weight.shape[0] # vocab_size
        self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, use_fast=False)
        
        # Eagle3 model 核心属性
        ea_config = EConfig.from_pretrained(ea_model_path) # 只能解析预先定义的字段
        # 读取bias配置
        with open(ea_model_path, "r") as f: # bias是一个自定义配置项
            ea_config_dict = json.loads(f.read())
        bias = ea_config_dict.get("bias", True)
        # 只是命名上是一个layer，其实是一个完整的子模型
        self.ea_layer = Model(config=ea_config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                            threshold=threshold, path=base_model_path,load_emb=True)
        
        # 设备/数据类型对齐
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device # target model 的最后一层的位置
        self.ea_layer.diff_device = (device != base_model.lm_head.weight.device)
        if self.ea_layer.diff_device:
            self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
        
        # 如果draft词表和原词表一致，就不需要词表映射
        if ea_config.vocab_size==ea_config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        
        # 加载Eagle3权重
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        # 对齐数据类型+设备
        self.ea_layer.to(self.base_model.dtype).to(device)
        # Eagle3 初始化树结构
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model."""
        return self.tokenizer


    # 一步到位，一口气 加载base model、eagle model、自动选择最优total token
    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
        ):
        # 需要定制化的，针对树的KV Cache结构，因此需要重新写modeling_llama_kv.py之类
        base_model_config = AutoConfig.from_pretrained(base_model_path)
        model_arch = base_model_config.architectures[0] # 读取模型架构名
        model_mapping = { # 映射到对应的KV Cache模型类，避免写大量if-else
            'LlamaForCausalLM': KVLlamaForCausalLM,
            'Qwen2ForCausalLM': KVQwen2ForCausalLM,
            'Qwen3ForCausalLM': KVQwen3ForCausalLM,
            'MixtralForCausalLM': KVMixtralForCausalLM,
        }
        # 提前抛出不支持的架构，避免奔溃
        if model_arch not in model_mapping:
            raise ValueError(f"Unsupported model architecture: {model_arch}")
        base_model = model_mapping[model_arch].from_pretrained(base_model_path, **kwargs)

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            raise FileNotFoundError(f"config.json not found in local directory: {ea_model_path}")

        # 加载Eagle3权重（分片、单文件、safetensors）
        ea_layer_state_dict = {}
        weight_dir = os.path.join(ea_model_path, "pytorch_model.bin")
        if os.path.isdir(weight_dir): # 分片权重
            shard_files = sorted([f for f in os.listdir(weight_dir) if f.endswith('.bin')])
            if not shard_files:
                raise FileNotFoundError(f"No .bin shards found in {weight_dir}")
            for shard_file in shard_files:
                shard_path = os.path.join(weight_dir, shard_file)
                shard_state_dict = torch.load(shard_path, map_location=base_model.device)
                ea_layer_state_dict.update(shard_state_dict)
            print(f"✅ Loaded {len(shard_files)} weight shards from {weight_dir}")
        elif os.path.isfile(weight_dir): # 单片权重
            ea_layer_state_dict = torch.load(weight_dir, map_location=base_model.device)
            print(f"✅ Loaded single weight file from {weight_dir}")
        else: # 兜底：尝试 safetensors
            from safetensors.torch import load_file
            weight_path = os.path.join(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(weight_path)
            print(f"✅ Loaded safetensors file from {weight_path}")

        #初始化模型
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        # total_token定义了「草稿模型（ea_layer）一次生成多少个候选 token
        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            candidate_tokens = [40, 48, 50, 56, 60]
            time_factors = [1, 1.05, 1.07, 1.1, 1.13]
            infer_times = []
            
            for idx, token_num in enumerate(candidate_tokens):
                # 基准测试：随机输入+多次推理
                input_ids = torch.randint(0,model.config.vocab_size - 200, (1,token_num)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    with torch.no_grad():
                        model.base_model(input_ids)
                    torch.cuda.synchronize()
                end_time = time.time()
                infer_times.append((end_time - start_time) / time_factors[idx])

            #  选择耗时最小的token数
            best_token = candidate_tokens[infer_times.index(min(infer_times))]
            # base_model 的输入长度需要留 1 个位置给「验证后的真实 token」，减 1 后避免输入长度超过模型的 max_position_embeddings
            model.ea_layer.total_tokens = best_token - 1 
        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        """前向传播，仅使用target model，返回隐藏层/原始logits"""
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            hidden_states = outputs[0] # 为什么取[0]? outputs 元组有哪些项？ 
            if output_orig: # 这里orig是什么东西？为什么会有outputorig？
                orig = self.base_model.lm_head(outputs[0])
                return outputs, orig, hidden_states
            return outputs, hidden_states

    # Eagle3 加速生成（一次性返回所有token）
    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

        ):
        # 初始化停止符号，兼容Llama3的<|eot_id|>
        stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if is_llama3 else None
        # 初始化logits处理器（采样/贪心）
        logits_processor = prepare_logits_processor(temperature, top_p, top_k) if temperature > 1e-5 else None

        # 初始化输入/缓存
        input_ids = input_ids.clone()
        # 生成一个值为 -1 的张量作为「占位符 / 填充 token」，用于 Eagle3 树结构中未填充的位置
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        self.ea_layer.reset_kv()
        # 模型输入的 token id 张量，形状为 (batch_size, seq_len)（此处 batch_size 固定为 1）
        input_len = input_ids.shape[1]

        # 初始化KV Cache
        if hasattr(self, "past_key_values"): # 检查对象obj是否具有属性past_key_values，返回布尔值
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # 预填充：初始化Eagle3树结构
        reset_tree_mode(self)
        (
            draft_tokens, 
            retrieve_indices, 
            tree_mask, 
            tree_position_ids, 
            logits, 
            hidden_state, 
            sample_token,
            ) = initialize_tree(input_ids, self, past_key_values, logits_processor)
        new_token = 0
        # self.ea_layer.total_tokens 是 Eagle3 树结构的最大候选 token 数；
        # 减 10 是预留冗余空间，避免树结构扩展时超出 KV Cache 的最大长度限制
        max_gen_length = max_length - self.ea_layer.total_tokens - 10

        # Eagle3核心生成循环
        for idx in range(max_gen_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            
            # 树解码，批量生成候选token
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            
            # 验证候选token，选择最优序列
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # 更新输入和树结构 
            (
                input_ids, 
                draft_tokens, 
                retrieve_indices, 
                tree_mask, 
                tree_position_ids, 
                new_token, 
                hidden_state, 
                sample_token, 
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )
            
            # 终止条件判断
            gen_tokens = input_ids[0, input_len:].tolist()
            # 判断llama3的stop_token
            llama3_stop_condition = (stop_token_id is not None and stop_token_id in gen_tokens)
            # 判断通用eos_token
            common_stop_condition = (self.tokenizer.eos_token_id in gen_tokens)
            if llama3_stop_condition or common_stop_condition:
                break
            if new_token > max_new_tokens or input_ids.shape[1] > max_gen_length:
                break
 
        return input_ids if not log else (input_ids, new_token, idx)
    
    # 原生生成，一次性返回
    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

        ):
        stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if is_llama3 else None
        logits_processor = prepare_logits_processor(temperature, top_p, top_k) if temperature > 1e-5 else None

        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        input_len = input_ids.shape[1]

        # 初始化KV Cache
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()  # 重置长度计数器
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # 预填充
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        # 为了避免溢出，因为已经读取了eagle3，其实后面可以搞的不读取这样。
        max_gen_length = max_length - self.ea_layer.total_tokens - 10  

        for idx in range(max_gen_length):
            # 贪心采样选择token
            if logits_processor:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1
            '''
            Token（令牌）：文本被切分后的最小单位（如单词 / 子词，例："你好" → ["你", "好"]）；
            Token ID：每个 Token 对应的整数编码（由 tokenizer 映射，例："你" → 123，"好" → 456）；
            Embedding（嵌入）：模型将 Token ID 转为高维向量（如 768 维），是模型计算的输入；
            self.base_model(input_ids, ...) 内部会调用 embedding layer，将 Token ID 转为 embedding 向量，无需手动处理（这是 Transformer的标准流程）。
            '''

            # 终止条件
            gen_tokens = input_ids[0, input_len:].tolist()
            # 判断llama3的stop_token
            llama3_stop_condition = (stop_token_id is not None and stop_token_id in gen_tokens)
            # 判断通用eos_token
            common_stop_condition = (self.tokenizer.eos_token_id in gen_tokens)
            if llama3_stop_condition or common_stop_condition:
                break
            if new_token > max_new_tokens or input_ids.shape[1] > max_gen_length:
                break
        return input_ids if not log else (input_ids, new_token, idx)

    # Eagle3 流式返回
    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

        ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

        ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
