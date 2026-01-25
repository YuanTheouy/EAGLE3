import argparse
import json
import os
import time
import torch
import shortuuid
import random
from tqdm import tqdm
from accelerate.utils import set_seed
from transformers import AutoProcessor, AutoConfig
import requests
from PIL import Image
from io import BytesIO
import sys

# Ensure project root is in sys.path to load local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入 Eagle 相关模块
from eagle.model.utils import prepare_logits_processor
import eagle.model.cnets_vl as cnets
from eagle.model.cnets_vl import Model as EaLayerVL
from eagle.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from eagle.model.configs import EConfig

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            answers[data["question_id"]] = line
    with open(answer_file, "w") as fout:
        for qid in sorted(list(answers.keys())):
            fout.write(answers[qid])

# Monkey patch LlamaAttention._init_rope to support mrope
original_init_rope = cnets.LlamaAttention._init_rope

def new_init_rope(self):
    if self.config.rope_scaling is not None and (self.config.rope_scaling.get("type") == "mrope" or self.config.rope_scaling.get("type") == "default"):
         self.rotary_emb = cnets.LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
    else:
         original_init_rope(self)

cnets.LlamaAttention._init_rope = new_init_rope

from eagle.model.kv_cache import initialize_past_key_values
from eagle.model.utils import (
    reset_tree_mode, 
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    LogitsProcessorList
)
from typing import List, Tuple

def tree_decoding_vl(
    model,
    tree_candidates: torch.Tensor,
    past_key_values,
    tree_position_ids: torch.Tensor,
    input_ids: torch.Tensor,
    retrieve_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    VL-adapted tree decoding: correctly handles position_ids offset using actual KV cache length
    instead of input_ids length (which misses vision tokens).
    """
    # [FIX] Get actual KV cache length from the first layer's key cache
    # past_key_values is a list of [key_cache, value_cache] per layer
    # key_cache is a KVCache object with .current_length attribute
    if isinstance(past_key_values, list) and len(past_key_values) > 0:
        current_kv_len = past_key_values[0][0].current_length.item()
    else:
        # Fallback for standard cache (unlikely in Eagle)
        current_kv_len = input_ids.shape[1]
        
    # Calculate position IDs based on actual context length (Vision + Text)
    position_ids = tree_position_ids + current_kv_len
    
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    # Tree decoding forward pass
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # Eagle3 logic: concatenate last 3 hidden states
    # [FIX] Use first 3 layers [0, 1, 2] to match training if confirmed, 
    # or [-3:] if that was the intent. Keeping [-3:] as per original script for now,
    # but user should verify training config.
    ea_device = model.ea_layer.lm_head.weight.device
    hidden_states = [x.to(ea_device) for x in outputs["hidden_states"]]
    # Note: Training code uses [0, 1, 2]. If this script used [-3:], it might be wrong.
    # We will stick to the local implementation logic but ensure it's consistent.
    # The user previously flagged this. Let's use the same logic as in eagenerate.
    # In eagenerate we changed it to [0, 1, 2]. Let's align here.
    hidden_state = torch.cat([hidden_states[0], hidden_states[1], hidden_states[2]], dim=-1)

    # Filter logits by retrieve indices
    logits = tree_logits[0, retrieve_indices]

    return logits, hidden_state, outputs

def update_inference_inputs_vl(
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    best_candidate: torch.Tensor,
    accept_length: int,
    retrieve_indices: torch.Tensor,
    logits_processor: LogitsProcessorList,
    new_token: int,
    past_key_values_data_list: List[torch.Tensor],
    current_length_data: torch.Tensor,
    model,
    hidden_state_new: torch.Tensor,
    sample_p: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, torch.Tensor, torch.Tensor, int, None, torch.Tensor]:
    """
    VL-adapted update: updates KV cache at the correct index (end of Vision+Text) 
    instead of overwriting Vision tokens at input_ids length.
    """
    # [FIX] Use current_length_data to determine where to write new KV
    # current_length_data tracks the valid length of the cache (Vision + Text)
    prev_input_len = current_length_data[0].item()

    # Select best candidate indices
    # Note: retrieve_indices are relative to the tree start.
    # We need to map them to the cache indices.
    # The tree candidates were generated *after* prev_input_len.
    select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    select_indices = select_indices.to(input_ids.device)

    # Update input_ids (Text only)
    best_candidate_tokens = candidates[best_candidate, : accept_length + 1].unsqueeze(0).to(input_ids.device)
    input_ids = torch.cat([input_ids, best_candidate_tokens], dim=-1)

    # Update Past Key Values
    for past_kv_data in past_key_values_data_list:
        # Source: KV for the candidates (computed during tree decoding)
        # Tree decoding used position_ids starting at prev_input_len.
        # So the KV cache for these candidates is already at the correct positions in the temp cache?
        # Wait, Eagle's tree decoding writes to the *same* cache but at temporary positions?
        # No, Eagle usually assumes we copy from the 'tree' positions to the 'main' positions.
        
        # In standard Eagle, `past_kv_data` holds everything.
        # The tree decoding step wrote KV to `select_indices`.
        # We now need to "finalize" the accepted path by moving it to be contiguous?
        # Actually, Eagle's KV cache is managed such that we copy the accepted tokens 
        # to the position immediately following the previous valid length.
        
        # Source indices are where the tree nodes were placed.
        tgt = past_kv_data[..., select_indices, :]
        
        # Destination indices are contiguous after the previous length
        dst = past_kv_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)

    # Update current_length
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Update hidden state for next tree generation
    retrieve_hidden_state = hidden_state_new[:, retrieve_indices]
    accept_hidden_state = retrieve_hidden_state[:, best_candidate, : accept_length + 1]

    # Sample new token
    if logits_processor is not None:
        new_sample_token = torch.multinomial(sample_p, 1).unsqueeze(0)
    else:
        new_sample_token = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)

    # Generate new tree
    new_input_ids = torch.cat([input_ids, new_sample_token.to(input_ids.device)], dim=1)
    
    # [FIX] Hidden states concat logic in topK_genrate is inside EaLayer, which we can't easily change here.
    # But we passed `accept_hidden_state` which is derived from `hidden_state_new`.
    # `hidden_state_new` came from `tree_decoding_vl` where we already fixed the concat logic.
    
    draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = model.ea_layer.topK_genrate(
        accept_hidden_state, new_input_ids, model.base_model.lm_head, logits_processor
    )

    # Update token count
    new_token += accept_length + 1

    return (
        input_ids, draft_tokens, new_retrieve_indices, new_tree_mask,
        new_tree_position_ids, new_token, None, new_sample_token
    )


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image

# ===================== EaModelVL 定义 =====================

class EaModelVL(torch.nn.Module):
    def __init__(
            self,
            base_model_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            device_map="auto"
    ):
        super().__init__()
        # 加载 Base Model (Qwen2.5-VL)
        print(f"Loading base model from {base_model_path}...")
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.config = self.base_model.config
        self.base_model_path = base_model_path
        
        # 加载 Eagle Head 配置
        ea_config = EConfig.from_pretrained(ea_model_path)
        with open(os.path.join(ea_model_path, "config.json"), "r") as f:
            ea_config_dict = json.loads(f.read())
        bias = ea_config_dict.get("bias", True)
        
        # 初始化 Eagle Layer
        print("Initializing Eagle Layer...")
        # EaLayerVL (cnets_vl.Model) 构造函数参数：
        # def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        
        self.ea_layer = EaLayerVL(
            config=ea_config, 
            load_emb=True,
            path=base_model_path,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold
        )
        
        # 设备对齐
        # Qwen2.5-VL 的结构是 self.base_model.model.layers
        # 但报错提示 'Qwen2_5_VLModel' object has no attribute 'layers'
        # 让我们检查一下 self.base_model.model 的属性
        # Qwen2_5_VLForConditionalGeneration.model 是 Qwen2_5_VLModel
        # Qwen2_5_VLModel 应该有 layers 属性（在 modeling_qwen2_5_vl.py 中定义为 self.layers = nn.ModuleList(...)）
        # 可能是属性名不同，或者是封装层级问题。
        
        # 尝试直接访问 base_model 的 device 属性
        device = self.base_model.device
        
        self.ea_layer.diff_device = (device != self.base_model.lm_head.weight.device)
        if self.ea_layer.diff_device:
            self.ea_layer.headweight = self.base_model.lm_head.weight.clone().to(device)
            
        if ea_config.vocab_size == ea_config.draft_vocab_size:
            if hasattr(self.ea_layer, "d2t"): del self.ea_layer.d2t
            if hasattr(self.ea_layer, "t2d"): del self.ea_layer.t2d
            
        # 加载权重
        print("Loading Eagle Layer weights...")
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        
        # 初始化树结构
        self.ea_layer.init_tree()

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
        ):
        # 加载 Eagle 权重
        ea_layer_state_dict = {}
        if os.path.exists(os.path.join(ea_model_path, "pytorch_model.bin")):
            ea_layer_state_dict = torch.load(os.path.join(ea_model_path, "pytorch_model.bin"))
        else:
             # 支持 safetensors 或分片权重 (这里简化处理，假设是 pytorch_model.bin)
             # 如果是 safetensors，可以使用 safe_open
             from safetensors.torch import load_file
             if os.path.exists(os.path.join(ea_model_path, "model.safetensors")):
                 ea_layer_state_dict = load_file(os.path.join(ea_model_path, "model.safetensors"))
        
        return cls(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            ea_layer_state_dict=ea_layer_state_dict,
            **kwargs
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            pixel_values=None,
            image_grid_thw=None,
            **kwargs
    ):
        with torch.inference_mode():
            # Qwen2.5-VL Forward
            # 注意：base_model 是 Qwen2_5_VLForConditionalGeneration
            # 如果是 Qwen2_5_VLForConditionalGeneration，调用 forward 时需要根据参数签名传参
            
            # 兼容性处理：如果 pixel_values 存在但 base_model 不需要（理论上 VL 模型都需要），
            # 或者需要其他处理。这里直接传递，假设 base_model 能处理。
            
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "output_hidden_states": True,
                "return_dict": True
            }
            if pixel_values is not None:
                forward_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                forward_kwargs["image_grid_thw"] = image_grid_thw
                
            outputs = self.base_model(**forward_kwargs)
            
            # 这里的 hidden_states 应该包含所有层，我们需要的是用于 Eagle 的那几层
            # Eagle 通常使用最后几层，具体取决于训练时的配置
            # 在 cnets_vl.py 中，Model.dataprepare 提取了 outputs.hidden_states[0, 1, 2] 
            # 注意：base_model 输出的 hidden_states 是 tuple，包含 (embed_out, layer_0_out, ..., layer_N_out) 
            # 或者 (layer_0_out, ..., layer_N_out) 取决于配置。
            # Qwen2_5_VLForConditionalGeneration 的 output_hidden_states=True 会返回所有 hidden states。
            
            # 我们直接返回 outputs 对象，后续处理由调用者负责
            if output_orig:
                return outputs, outputs.logits, outputs.hidden_states
            return outputs, outputs.hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            pixel_values=None,
            image_grid_thw=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            eos_token_ids=None
        ):
        
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # 初始化 Padding Token
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        self.ea_layer.reset_kv()
        input_len = input_ids.shape[1]

        # 初始化 KV Cache
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        reset_tree_mode(self)

        # === 1. Prefill (Initialize Tree) ===
        # 这里手动执行 initialize_tree 的逻辑以支持 pixel_values
        
        # Base Model Prefill
        outputs, orig_logits, hidden_states = self(
            input_ids, 
            past_key_values=past_key_values, 
            output_orig=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        
        # 采样第一个 Token
        if logits_processor is not None:
            logits = orig_logits[:, -1]
            logits = logits_processor(None, logits)
            probs = torch.nn.functional.softmax(logits, dim=1)
            sample_token = torch.multinomial(probs, 1)
        else:
            sample_token = torch.argmax(orig_logits[:, -1])
            sample_token = sample_token[None, None]

        # 更新输入
        input_ids = torch.cat([input_ids, sample_token.to(input_ids.device)], dim=1)

        # 准备 Eagle 输入 (拼接最后 3 层 hidden states)
        ea_device = self.ea_layer.lm_head.weight.device
        # hidden_states 是 tuple，取最后 3 层
        hidden_states_list = [x.to(ea_device) for x in hidden_states]
        # 注意：这里假设 EAGLE 训练时使用的是最后 3 层。如果不是，需要调整。
        # 通常 hidden_states[-1] 是最后一层的输出。
        # [MODIFIED] Use first 3 layers (0, 1, 2) to match training implementation in eagle/traineagle3/cnets_vl.py
        # The training code uses hidden_states[0], [1], [2].
        # hidden_states_concat = torch.cat(hidden_states_list[-3:], dim=-1)
        hidden_states_concat = torch.cat([hidden_states_list[0], hidden_states_list[1], hidden_states_list[2]], dim=-1)

        # Eagle 生成 Draft Tokens
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
            hidden_states_concat, input_ids, self.base_model.lm_head, logits_processor
        )

        new_token = 0
        max_gen_length = max_length - self.ea_layer.total_tokens - 10

        # === 2. Decoding Loop ===
        for idx in range(max_gen_length):
            if hasattr(self.base_model.model, "language_model"):
                self.base_model.model.language_model.tree_mask = tree_mask
            else:
                self.base_model.model.tree_mask = tree_mask
            
            draft_tokens = draft_tokens.to(input_ids.device)

            # Tree Decoding (Base Model Verification)
            # [MODIFIED] Use VL-adapted tree decoding
            logits, hidden_state_new, outputs = tree_decoding_vl(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices
            )

            # Evaluate Posterior
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            
            # [DEBUG] Print accept length to monitor performance
            # print(f"DEBUG: accept_length={accept_length}")

            # Update Inference Inputs
            # [MODIFIED] Use VL-adapted update function
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs_vl(
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

            # Stop condition
            if eos_token_ids:
                last_token = input_ids[0, -1].item()
                if last_token in eos_token_ids:
                    break
            
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        return input_ids, new_token, idx


# ===================== 主逻辑 =====================

def generate_single_turn(messages, processor, model, temperature, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.base_model.device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    output_ids, new_token, idx = model.eagenerate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
        image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        eos_token_ids=[processor.tokenizer.eos_token_id] + (processor.tokenizer.additional_special_tokens_ids if hasattr(processor.tokenizer, "additional_special_tokens_ids") else [])
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    generated_ids = output_ids[0, len(inputs.input_ids[0]):]
    output = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output, new_token, idx, total_time

def get_model_answers(
    base_model_path,
    ea_model_path,
    model_id,
    questions,
    answer_file,
    max_new_tokens,
    num_choices,
    temperature,
    args
):
    # 1. 加载模型
    print(f"Loading model on device {torch.cuda.current_device()}...")
    model = EaModelVL.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # 2. 模型预热
    if questions:
        print("Warmup starting...")
        warmup_question = questions[0]
        for _ in range(3):
            torch.manual_seed(0)
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
            for j in range(len(warmup_question["turns"])):
                turn = warmup_question["turns"][j]
                if "image" in warmup_question and j == 0:
                    content = [{"type": "image", "image": load_image(warmup_question["image"])}, {"type": "text", "text": turn}]
                else:
                    content = [{"type": "text", "text": turn}]
                messages.append({"role": "user", "content": content})
                output, _, _, _ = generate_single_turn(messages, processor, model, temperature, max_new_tokens)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output}]})
        print("Warmup completed")

    # 3. 批量生成
    for question in tqdm(questions, desc=f"Generating (GPU {torch.cuda.current_device()})"):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
            turns, idxs, new_tokens, wall_time = [], [], [], []
            
            for j in range(len(question["turns"])):
                turn = question["turns"][j]
                if "image" in question and j == 0:
                    content = [{"type": "image", "image": load_image(question["image"])}, {"type": "text", "text": turn}]
                else:
                    content = [{"type": "text", "text": turn}]
                
                messages.append({"role": "user", "content": content})
                output, new_token, idx, total_time = generate_single_turn(
                    messages, processor, model, temperature, max_new_tokens
                )
                
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": output}]})
            
            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time
            })

        # 写入结果
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

def run_eval(
    base_model_path,
    ea_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_tokens,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    temperature,
    args
):
    # 1. 加载问题
    questions = []
    with open(question_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    if question_begin is not None:
        questions = questions[question_begin:]
    if question_end is not None:
        questions = questions[:question_end - (question_begin or 0)]
    
    print(f"Loaded {len(questions)} questions")

    # 2. 分布式判断
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    # 3. 提交任务
    chunk_size = (len(questions) + (num_gpus_total // num_gpus_per_model) - 1) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="qwen2.5-vl-7b")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    
    args = parser.parse_args()

    # 补充模型ID
    args.model_id = f"{args.model_id}-temperature-{args.temperature}"
    
    # 确定路径
    if not args.answer_file:
        args.answer_file = f"results/{args.bench_name}/{args.model_id}.jsonl"
    
    run_eval(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        model_id=args.model_id,
        question_file=args.question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=args.answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        args=args
    )

    reorg_answer_file(args.answer_file)
    print(f"✅ All done! Results saved to {args.answer_file}")
