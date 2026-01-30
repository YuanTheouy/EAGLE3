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
    update_inference_inputs
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
            
        # 检查是否为小词表模式
        if ea_config.vocab_size == ea_config.draft_vocab_size:
            print("Note: Draft vocab size equals Base vocab size. No clustering used.")
            if hasattr(self.ea_layer, "d2t"): del self.ea_layer.d2t
            if hasattr(self.ea_layer, "t2d"): del self.ea_layer.t2d
        else:
            print(f"Note: Using Small Vocab Clustering. Base: {ea_config.vocab_size}, Draft: {ea_config.draft_vocab_size}")
            # 必须确保 d2t/t2d 被正确加载
            if hasattr(self.ea_layer, "d2t"):
                # 检查 d2t 是否全为 0 (意味着未加载或初始化失败)
                if self.ea_layer.d2t.sum() == 0:
                    print("CRITICAL WARNING: d2t (Draft-to-Token map) is all zeros! Weights likely not loaded for buffers.")
                    # 尝试从 state_dict 中手动查找并加载
                    for k, v in new_state_dict.items():
                        if "d2t" in k:
                            print(f"Found d2t in weights: {k}, shape: {v.shape}")
                            self.ea_layer.d2t.data = v.to(device)
                        if "t2d" in k:
                             self.ea_layer.t2d.data = v.to(device)
                    
                    if self.ea_layer.d2t.sum() == 0:
                        print("ERROR: Failed to load d2t mapping! This will cause 100% rejection rate.")

        # 加载权重
        print("Loading Eagle Layer weights...")
        
        # 处理可能的 Key 前缀不匹配问题
        new_state_dict = {}
        for k, v in ea_layer_state_dict.items():
            # 去除可能的 module. 前缀 (DDP)
            if k.startswith("module."):
                k = k[7:]
            # 去除可能的 base_model. 前缀 (如果在某些训练框架下保存)
            if k.startswith("base_model."):
                k = k[11:]
            new_state_dict[k] = v

        # 移动到这里，确保 new_state_dict 已经准备好
        # 检查是否为小词表模式
        if ea_config.vocab_size == ea_config.draft_vocab_size:
            print("Note: Draft vocab size equals Base vocab size. No clustering used.")
            if hasattr(self.ea_layer, "d2t"): del self.ea_layer.d2t
            if hasattr(self.ea_layer, "t2d"): del self.ea_layer.t2d
        else:
            print(f"Note: Using Small Vocab Clustering. Base: {ea_config.vocab_size}, Draft: {ea_config.draft_vocab_size}")
            # 必须确保 d2t/t2d 被正确加载
            if hasattr(self.ea_layer, "d2t"):
                # 先加载权重（包括 buffers）
                missing_keys, unexpected_keys = self.ea_layer.load_state_dict(new_state_dict, strict=False)
                
                # 检查 d2t 是否全为 0
                if self.ea_layer.d2t.sum() == 0:
                     print("CRITICAL WARNING: d2t is all zeros after load_state_dict!")
                     # 再次尝试手动加载（双重保险）
                     for k, v in new_state_dict.items():
                        if "d2t" in k:
                            print(f"Found d2t in weights: {k}, shape: {v.shape}")
                            self.ea_layer.d2t.data = v.to(device)
                        if "t2d" in k:
                             self.ea_layer.t2d.data = v.to(device)

        
        # 统一执行加载（覆盖上面的逻辑，确保 missing_keys 变量存在）
        missing_keys, unexpected_keys = self.ea_layer.load_state_dict(new_state_dict, strict=False)
        
        # 再次检查 d2t (如果在小词表模式下)
        if ea_config.vocab_size != ea_config.draft_vocab_size and hasattr(self.ea_layer, "d2t"):
             if self.ea_layer.d2t.sum() == 0:
                  print("CRITICAL WARNING: d2t is still zero after unified load_state_dict!")
        
        if len(missing_keys) > 0:
            # 过滤掉 embed_tokens.weight，因为我们已经手动复制了
             filtered_missing = [k for k in missing_keys if "embed_tokens.weight" not in k]
             
             # 如果是小词表模式，d2t/t2d 必须存在
             if ea_config.vocab_size != ea_config.draft_vocab_size:
                 if "d2t" in missing_keys or "t2d" in missing_keys:
                     print("CRITICAL ERROR: d2t or t2d mapping buffers are missing from checkpoint! Small vocab clustering cannot work.")
             
             if len(filtered_missing) > 0:
                print(f"WARNING: Missing keys in Eagle Layer: {filtered_missing[:5]} ... (Total {len(filtered_missing)})")
                # 关键：检查核心权重是否缺失
                if "midlayer.layers.0.self_attn.q_proj.weight" in filtered_missing or "fc.weight" in filtered_missing:
                    print("CRITICAL WARNING: Core Eagle weights are missing! Performance will be degraded to random guessing.")
        
        if len(unexpected_keys) > 0:
            print(f"WARNING: Unexpected keys in Eagle weights: {unexpected_keys[:5]} ... (Total {len(unexpected_keys)})")
            
        self.ea_layer.to(self.base_model.dtype).to(device)
        
        # 验证 Embedding Norm
        eagle_emb_norm = self.ea_layer.embed_tokens.weight.norm().item()
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
             base_emb_norm = self.base_model.model.embed_tokens.weight.norm().item()
             print(f"Embedding Norm Check - Base: {base_emb_norm:.4f}, Eagle: {eagle_emb_norm:.4f}")
        else:
             print(f"Eagle Embedding Norm: {eagle_emb_norm:.4f}")
        
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
        ea_weight_path = os.path.join(ea_model_path, "pytorch_model.bin")
        if os.path.exists(ea_weight_path):
            print(f"Loading Eagle weights from {ea_weight_path}")
            ea_layer_state_dict = torch.load(ea_weight_path, map_location='cpu')
        else:
             # 支持 safetensors 或分片权重
             ea_weight_path = os.path.join(ea_model_path, "model.safetensors")
             if os.path.exists(ea_weight_path):
                 print(f"Loading Eagle weights from {ea_weight_path}")
                 from safetensors.torch import load_file
                 ea_layer_state_dict = load_file(ea_weight_path, device='cpu')
             else:
                 # 尝试 adapter_model.bin (LoRA 风格)
                 ea_weight_path = os.path.join(ea_model_path, "adapter_model.bin")
                 if os.path.exists(ea_weight_path):
                     print(f"Loading Eagle weights from {ea_weight_path}")
                     ea_layer_state_dict = torch.load(ea_weight_path, map_location='cpu')
                 else:
                     print(f"WARNING: No Eagle weights found in {ea_model_path}. Please check your path!")
        
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
            # [MODIFIED] Inline tree_decoding logic to ensure first 3 layers are used for hidden_state_new
            position_ids = tree_position_ids + input_ids.shape[1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

            # 树解码前向传播
            outputs, tree_logits, hidden_states = self(
                draft_tokens,
                output_orig=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            # Eagle3专属：拼接前3层hidden states (与训练保持一致)
            ea_device = self.ea_layer.lm_head.weight.device
            hidden_states_list = [x.to(ea_device) for x in hidden_states]
            hidden_state_new = torch.cat(hidden_states_list[:3], dim=-1)

            # 按检索索引筛选logits
            logits = tree_logits[0, retrieve_indices]

            # Evaluate Posterior
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # Update Inference Inputs
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
