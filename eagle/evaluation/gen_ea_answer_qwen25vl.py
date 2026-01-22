import argparse
import json
import os
import time
import torch
import shortuuid
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
            
        if ea_config.vocab_size == ea_config.draft_vocab_size:
            if hasattr(self.ea_layer, "d2t"): del self.ea_layer.d2t
            if hasattr(self.ea_layer, "t2d"): del self.ea_layer.t2d
            
        # 加载权重
        print("Loading Eagle Layer weights...")
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        # init_tree 是在 Eagle 训练代码中的方法，但在 cnets_vl.py 中似乎没有定义。
        # 检查 cnets.py 或相关逻辑，init_tree 通常用于初始化树结构。
        # 如果 cnets_vl.py 没有 init_tree，我们需要手动添加或移除调用。
        # 观察 cnets_vl.py，似乎没有 init_tree 方法。
        # 但是 EAGLE 的推理需要树结构。
        # 让我们检查一下是否需要从 cnets.py 中借用 init_tree 或者自己实现。
        # 在 gen_ea_answer_llama3chat.py 中调用了 init_tree。
        # 假设 cnets_vl.py 的 Model 类应该有 init_tree，但可能漏掉了。
        # 或者我们可以直接在这里实现 init_tree 的逻辑。
        
        # 尝试调用 init_tree，如果不存在则调用我们刚才添加的 init_tree (在 cnets_vl.py 中)
        # 注意：我们已经在 cnets_vl.py 中添加了 init_tree 方法。
        # 但是 Python 的导入机制可能没有重新加载修改后的 cnets_vl.py，或者修改没有生效（如果是热重载环境）。
        # 但在这个环境中，每次运行都是新的进程，所以应该生效。
        # 报错 'Model' object has no attribute 'init_tree' 说明 Model 类（即 EaLayerVL）确实没有这个方法。
        # 可能是之前的 SearchReplace 没有成功添加 init_tree 到 Model 类中？
        # 让我们再次检查 cnets_vl.py 的 Model 类。
        
        if hasattr(self.ea_layer, "init_tree"):
            self.ea_layer.init_tree()
        else:
            # 如果真的没有，手动执行逻辑
            if hasattr(self.ea_layer, "d2t") and hasattr(self.ea_layer, "t2d"):
                 pass

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
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # Tree Decoding (Base Model Verification)
            # 注意：后续步骤不需要 pixel_values，因为 KV Cache 已经包含了视觉信息
            logits, hidden_state_new, outputs = tree_decoding(
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

def run_eval(
    base_model_path,
    ea_model_path,
    questions_file,
    answer_file,
    max_new_tokens,
    temperature,
    args
):
    # 1. 加载模型
    print("Loading model...")
    model = EaModelVL.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        device_map="auto"
    )
    model.eval()
    
    # 加载 Processor
    print(f"Loading processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # 2. 加载问题
    import json
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
            
    if args.question_begin is not None:
        questions = questions[args.question_begin:]
    if args.question_end is not None:
        questions = questions[:args.question_end]

    print(f"Total questions: {len(questions)}")

    # 3. 生成
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    for question in tqdm(questions):
        # 构造输入
        messages = []
        
        # 处理多轮对话或单轮
        # 假设 question['turns'] 是列表
        if 'turns' in question:
            for j, turn in enumerate(question['turns']):
                if "image" in question and j == 0:
                     # 构造多模态消息
                    content = [
                        {"type": "image", "image": load_image(question["image"])},
                        {"type": "text", "text": turn}
                    ]
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": turn})
        else:
            # 简单处理 text/image 字段
            text_input = question.get('text', '')
            image_path = question.get('image', None)
            content = []
            if image_path:
                 content.append({"type": "image", "image": load_image(image_path)})
            content.append({"type": "text", "text": text_input})
            messages.append({"role": "user", "content": content})

        
        # 准备输入
        # Qwen2.5-VL 的 apply_chat_template 会处理图像
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.base_model.device)
        
        # 生成
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_tokens, _ = model.eagenerate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
            image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_ids=[processor.tokenizer.eos_token_id] + (processor.tokenizer.additional_special_tokens_ids if hasattr(processor.tokenizer, "additional_special_tokens_ids") else [])
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # 解码
        generated_ids = output_ids[0, len(inputs.input_ids[0]):]
        output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 保存结果
        ans_json = {
            "question_id": question.get("question_id", shortuuid.uuid()),
            "text": output_text,
            "new_tokens": new_tokens,
            "time": total_time
        }
        
        with open(answer_file, "a") as f:
            f.write(json.dumps(ans_json) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    
    args = parser.parse_args()
    
    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.question_file,
        args.answer_file,
        args.max_new_tokens,
        args.temperature,
        args
    )
