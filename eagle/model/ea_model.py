import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


'''
整体是一个Eagle模型，包含一个基础模型和一个Eagle层
1、常量自定义区
2、初始化 
3、类方法 from_pretrained
4、前向传播 forward
    _prepare_generation 生成资源准备
    _check_termination 终止条件
5、生成方法
    EAGLE3生成 ea_generate / eagenerate
    普通生成 naive_generate / naivegenerate


'''


class EaModel(nn.Module):
    # Constants for generation
    SEQUENCE_LENGTH_BUFFER = 10  # Buffer for sequence length adjustment
    DEFAULT_TOTAL_TOKEN = 60      # Default total tokens for tree generation
    DEFAULT_DEPTH = 7             # Default depth of the tree
    DEFAULT_TOP_K = 10            # Default top-k value for sampling
    DEFAULT_THRESHOLD = 1.0       # Default threshold for posterior evaluation
    
    # Constants for automatic total_token selection
    TOTAL_TOKEN_CANDIDATES = [40, 48, 50, 56, 60]  # Candidate values for total_token
    TOTAL_TOKEN_ADJUSTMENT_FACTORS = [1, 1.05, 1.07, 1.1, 1.13]  # Adjustment factors for timing
    
    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d

        # # ========== 关键修复1：删除维度不匹配的lm_head.weight ==========
        # if "lm_head.weight" in ea_layer_state_dict:
        #     # 移除训练权重中32000维的lm_head，复用原生128256维的
        #     del ea_layer_state_dict["lm_head.weight"]
        #     print("✅ 删除训练权重中维度不匹配的lm_head.weight，复用Llama3.1原生lm_head")

        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def _prepare_generation(
            self,
            input_ids,
            temperature,
            top_p,
            top_k,
            max_length,
            is_llama3
    ):
        """
        Prepare common resources for generation methods.
        
        Args:
            input_ids: Input token IDs
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_length: Maximum sequence length
            is_llama3: Whether using Llama 3 model
            
        Returns:
            Dictionary containing prepared resources:
                - stop_token_id: Stop token ID for Llama 3
                - logits_processor: Logits processor for sampling
                - padding: Padding tensor
                - input_ids_cloned: Cloned input IDs to avoid in-place modification
                - past_key_values: Initialized past key values
                - past_key_values_data: Past key values data
                - current_length_data: Current length data
                - input_len: Original input length
                - adjusted_max_length: Adjusted maximum length
        """
        # Set up stop token ID for Llama 3
        stop_token_id = None
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        # Prepare logits processor
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        
        # Create padding tensor and clone input IDs
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids_cloned = input_ids.clone()
        
        # Reset kv cache
        self.ea_layer.reset_kv()
        
        # Initialize past key values
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
                self.base_model, max_length=max_length
            )
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        
        # Calculate input length and adjust max_length
        input_len = input_ids_cloned.shape[1]
        reset_tree_mode(self)
        adjusted_max_length = max_length - self.ea_layer.total_tokens - self.SEQUENCE_LENGTH_BUFFER
        
        return {
            "stop_token_id": stop_token_id,
            "logits_processor": logits_processor,
            "padding": padding,
            "input_ids_cloned": input_ids_cloned,
            "past_key_values": past_key_values,
            "past_key_values_data": past_key_values_data,
            "current_length_data": current_length_data,
            "input_len": input_len,
            "adjusted_max_length": adjusted_max_length
        }
    
    def _check_termination(
            self,
            input_ids,
            input_len,
            new_token,
            max_new_tokens,
            adjusted_max_length,
            is_llama3,
            stop_token_id
    ):
        """
        Check if generation should terminate based on various conditions.
        
        Args:
            input_ids: Current input token IDs
            input_len: Original input length
            new_token: Number of new tokens generated
            max_new_tokens: Maximum number of new tokens to generate
            adjusted_max_length: Adjusted maximum sequence length
            is_llama3: Whether using Llama 3 model
            stop_token_id: Stop token ID for Llama 3
            
        Returns:
            bool: True if generation should terminate, False otherwise
        """
        # Check for stop token (Llama 3 specific)
        if is_llama3 and stop_token_id is not None:
            if stop_token_id in input_ids[0, input_len:].tolist():
                return True
        
        # Check for EOS token
        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            return True
        
        # Check if max_new_tokens has been exceeded
        if new_token > max_new_tokens:
            return True
        
        # Check if sequence length exceeds adjusted_max_length
        if input_ids.shape[1] > adjusted_max_length:
            return True
        
        return False

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=None,
            depth=None,
            top_k=None,
            threshold=None,
            **kwargs,
    ):
        # Set default values if not provided
        if total_token is None:
            total_token = cls.DEFAULT_TOTAL_TOKEN
        if depth is None:
            depth = cls.DEFAULT_DEPTH
        if top_k is None:
            top_k = cls.DEFAULT_TOP_K
        if threshold is None:
            threshold = cls.DEFAULT_THRESHOLD
        import os
        import time
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModelForCausalLM

        # ========== 新增：校验 ea_model_path 是目录 ==========
        if not os.path.isdir(ea_model_path):
            raise ValueError(f"ea_model_path must be a directory, but got: {ea_model_path}")

        # ========== 原有逻辑：加载基础模型 ==========
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(base_model_path, **kwargs)

        # ========== 优化：加载 config.json（优先本地） ==========
        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            raise FileNotFoundError(f"config.json not found in local directory: {ea_model_path}")

        # ========== 核心修改：支持加载分片权重目录 ==========
        ea_layer_state_dict = {}
        # 权重目录：ea_model_path 下的 pytorch_model.bin 分片目录
        weight_dir = os.path.join(ea_model_path, "pytorch_model.bin")
        if os.path.isdir(weight_dir):
            # 遍历分片文件并合并
            shard_files = sorted([f for f in os.listdir(weight_dir) if f.endswith('.bin')])
            if not shard_files:
                raise FileNotFoundError(f"No .bin shards found in {weight_dir}")
            for shard_file in shard_files:
                shard_path = os.path.join(weight_dir, shard_file)
                shard_state_dict = torch.load(shard_path, map_location=base_model.device)
                ea_layer_state_dict.update(shard_state_dict)
            print(f"✅ Loaded {len(shard_files)} weight shards from {weight_dir}")
        elif os.path.isfile(weight_dir):
            # 兼容单个文件的情况
            ea_layer_state_dict = torch.load(weight_dir, map_location=base_model.device)
        else:
            # 兜底：尝试 safetensors（若有）
            from safetensors.torch import load_file
            weight_path = os.path.join(ea_model_path, "model.safetensors")
            if os.path.isfile(weight_path):
                ea_layer_state_dict = load_file(weight_path)
            else:
                raise FileNotFoundError(f"No weight files found in {ea_model_path}")

        # ========== 原有逻辑：初始化模型 ==========
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        # ========== 原有逻辑：自动选择 total_token ==========
        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = cls.TOTAL_TOKEN_CANDIDATES
            x = cls.TOTAL_TOKEN_ADJUSTMENT_FACTORS
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=self.use_eagle3,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

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
        # Prepare common generation resources
        generation_resources = self._prepare_generation(
            input_ids, temperature, top_p, top_k, max_length, is_llama3
        )
        
        stop_token_id = generation_resources["stop_token_id"]
        logits_processor = generation_resources["logits_processor"]
        padding = generation_resources["padding"]
        input_ids = generation_resources["input_ids_cloned"]
        past_key_values = generation_resources["past_key_values"]
        past_key_values_data = generation_resources["past_key_values_data"]
        current_length_data = generation_resources["current_length_data"]
        input_len = generation_resources["input_len"]
        adjusted_max_length = generation_resources["adjusted_max_length"]
        
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        
        for idx in range(adjusted_max_length):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            
            # Adjusting the input sequence, draft model forward
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

            # Check if generation should terminate
            if self._check_termination(
                    input_ids, input_len, new_token, max_new_tokens, 
                    adjusted_max_length, is_llama3, stop_token_id
            ):
                break
        
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

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
        # Prepare common generation resources
        generation_resources = self._prepare_generation(
            input_ids, temperature, top_p, top_k, max_length, is_llama3
        )
        
        stop_token_id = generation_resources["stop_token_id"]
        logits_processor = generation_resources["logits_processor"]
        input_ids = generation_resources["input_ids_cloned"]
        past_key_values = generation_resources["past_key_values"]
        input_len = generation_resources["input_len"]
        adjusted_max_length = generation_resources["adjusted_max_length"]
        
        # Initial forward pass
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        
        for idx in range(adjusted_max_length):
            # Generate next token
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            
            # Update outputs and input_ids
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            # Check if generation should terminate
            if self._check_termination(
                    input_ids, input_len, new_token, max_new_tokens, 
                    adjusted_max_length, is_llama3, stop_token_id
            ):
                break
        
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

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
        # Prepare common generation resources
        generation_resources = self._prepare_generation(
            input_ids, temperature, top_p, top_k, max_length, is_llama3
        )
        
        stop_token_id = generation_resources["stop_token_id"]
        logits_processor = generation_resources["logits_processor"]
        padding = generation_resources["padding"]
        input_ids = generation_resources["input_ids_cloned"]
        past_key_values = generation_resources["past_key_values"]
        past_key_values_data = generation_resources["past_key_values_data"]
        current_length_data = generation_resources["current_length_data"]
        input_len = generation_resources["input_len"]
        adjusted_max_length = generation_resources["adjusted_max_length"]
        
        # Initialize tree structure
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        
        for idx in range(adjusted_max_length):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            
            # Adjusting the input sequence, draft model forward
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

            # Check if generation should terminate
            if self._check_termination(
                    input_ids, input_len, new_token, max_new_tokens, 
                    adjusted_max_length, is_llama3, stop_token_id
            ):
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
        # Prepare common generation resources
        generation_resources = self._prepare_generation(
            input_ids, temperature, top_p, top_k, max_length, is_llama3
        )
        
        stop_token_id = generation_resources["stop_token_id"]
        logits_processor = generation_resources["logits_processor"]
        input_ids = generation_resources["input_ids_cloned"]
        past_key_values = generation_resources["past_key_values"]
        input_len = generation_resources["input_len"]
        adjusted_max_length = generation_resources["adjusted_max_length"]
        
        # Initial forward pass
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        
        for idx in range(adjusted_max_length):
            # Generate next token
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            
            # Update outputs and input_ids
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1
            
            # Yield the current sequence
            yield input_ids

            # Check if generation should terminate
            if self._check_termination(
                    input_ids, input_len, new_token, max_new_tokens, 
                    adjusted_max_length, is_llama3, stop_token_id
            ):
                break
    
    # Aliases for backward compatibility
    naivegenerate = naive_generate
    eagenerate = ea_generate