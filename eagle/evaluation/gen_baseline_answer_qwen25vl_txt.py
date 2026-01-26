"""Generate answers with Qwen2.5-VL model (compatible with EAGLE CLI)."""

import argparse
import json
import os
import sys
import time
import torch
import shortuuid
from tqdm import tqdm
from accelerate.utils import set_seed
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

# 固定随机种子（保证可复现）
set_seed(0)

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

# 辅助函数：加载图像
def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image

# 单轮对话生成
@torch.inference_mode()
def generate_single_turn(messages, processor, model, temperature, max_new_token):
    """
    Qwen2.5-VL适配的核心生成函数
    """
    # 处理消息格式
    processed_messages = []
    for msg in messages:
        if isinstance(msg["content"], str):
            content = [{"type": "text", "text": msg["content"]}]
        elif isinstance(msg["content"], list):
            converted_content = []
            for item in msg["content"]:
                if isinstance(item, str):
                    converted_content.append({"type": "text", "text": item})
                else:
                    converted_content.append(item)
            content = converted_content
        else:
            content = [{"type": "text", "text": str(msg["content"])}]
        
        processed_messages.append({
            "role": msg["role"],
            "content": content
        })
    
    # 编码输入
    inputs = processor.apply_chat_template(
        processed_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 模型生成
    torch.cuda.synchronize()
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_token,
        temperature=temperature if temperature > 1e-5 else 1e-5,
        top_p=0.95,
        do_sample=temperature > 1e-5,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # 解码输出
    output_ids = outputs[0][len(inputs.input_ids[0]):]
    stop_token_ids = [processor.tokenizer.eos_token_id]
    if stop_token_ids:
        stop_idx = [i for i, id in enumerate(output_ids) if id in stop_token_ids]
        if stop_idx:
            output_ids = output_ids[: stop_idx[0]]
    
    output = processor.batch_decode(
        [output_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    output = output.strip()
    
    new_token = len(output_ids)
    return output, new_token, total_time

# 主推理函数
def run_eval(args):
    # 1. 加载模型和处理器
    print(f"Loading Qwen2.5-VL model from {args.base_model_path}...")
    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        trust_remote_code=True
    )
    
    # 关键修复：使用专门的Qwen2.5-VL生成类
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={i: args.max_gpu_memory for i in range(torch.cuda.device_count())}
        )
        print("✓ Successfully loaded Qwen2_5_VLForConditionalGeneration")
    except Exception as e:
        print(f"Error loading Qwen2_5_VLForConditionalGeneration: {e}")
        raise
    
    model.eval()
    print("Model loaded successfully!")
    
    # 2. 加载问题数据集
    print(f"Loading questions from {args.question_file}...")
    questions = []
    with open(args.question_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    # 筛选问题范围
    if args.question_begin is not None:
        questions = questions[args.question_begin:]
    if args.question_end is not None:
        questions = questions[:args.question_end]
    
    print(f"Total questions to process: {len(questions)}")
    
    # 3. 创建输出目录
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    
    # 4. 批量生成答案
    for question in tqdm(questions, desc="Generating answers"):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        turns_output = []
        new_tokens_list = []
        times_list = []
        
        # 处理多轮对话
        if 'turns' in question:
            for j, turn in enumerate(question['turns']):
                if "image" in question and j == 0:
                    # 多模态输入
                    content = [
                        {"type": "image", "image": load_image(question["image"])},
                        {"type": "text", "text": turn}
                    ]
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": turn})
                
                # 生成回答
                output, new_token, total_time = generate_single_turn(
                    messages, processor, model, args.temperature, args.max_new_tokens
                )
                
                turns_output.append(output)
                new_tokens_list.append(new_token)
                times_list.append(total_time)
                
                # 添加助手回复到对话历史
                messages.append({"role": "assistant", "content": output})
        else:
            # 单轮输入（兼容不同格式）
            text_input = question.get('text', question.get('prompt', ''))
            image_path = question.get('image', None)
            
            content = []
            if image_path:
                try:
                    content.append({"type": "image", "image": load_image(image_path)})
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}")
            content.append({"type": "text", "text": text_input})
            
            messages.append({"role": "user", "content": content})
            
            # 生成回答
            output, new_token, total_time = generate_single_turn(
                messages, processor, model, args.temperature, args.max_new_tokens
            )
            
            turns_output.append(output)
            new_tokens_list.append(new_token)
            times_list.append(total_time)
        
        # 保存结果（对齐EAGLE输出格式）
        ans_json = {
            "question_id": question.get("question_id", shortuuid.uuid()),
            "text": turns_output[0] if len(turns_output) == 1 else turns_output,
            "new_tokens": sum(new_tokens_list),
            "time": sum(times_list),
            "model_id": f"qwen2.5vl-{args.temperature}",
            "choices": [{"index": 0, "turns": turns_output, "new_tokens": new_tokens_list, "wall_time": times_list}]
        }
        
        with open(args.answer_file, "a") as f:
            f.write(json.dumps(ans_json) + "\n")
    
    # 对答案文件去重并排序
    reorg_answer_file(args.answer_file)
    print(f"Answers saved to {args.answer_file}")

# 答案文件去重和排序
def reorg_answer_file(answer_file):
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            answers[data["question_id"]] = line
    
    with open(answer_file, "w") as fout:
        for qid in sorted(list(answers.keys())):
            fout.write(answers[qid])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 对齐EAGLE版本的核心参数
    parser.add_argument("--base-model-path", type=str, required=True, help="Path to Qwen2.5-VL base model")
    parser.add_argument("--ea-model-path", type=str, default="", help="Placeholder for compatibility with EAGLE CLI")
    parser.add_argument("--question-file", type=str, required=True, help="Path to input question file (JSONL)")
    parser.add_argument("--answer-file", type=str, required=True, help="Path to output answer file (JSONL)")
    parser.add_argument("--total-token", type=int, default=60, help="Placeholder for compatibility with EAGLE CLI")
    parser.add_argument("--depth", type=int, default=5, help="Placeholder for compatibility with EAGLE CLI")
    parser.add_argument("--top-k", type=int, default=10, help="Placeholder for compatibility with EAGLE CLI")
    
    # 原有核心参数
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--question-begin", type=int, default=None, help="Start index of questions")
    parser.add_argument("--question-end", type=int, default=None, help="End index of questions")
    parser.add_argument("--max-gpu-memory", type=str, default="80GiB", help="Max GPU memory per device")
    
    args = parser.parse_args()
    
    # 执行推理
    run_eval(args)
    print("Answer generation completed!")