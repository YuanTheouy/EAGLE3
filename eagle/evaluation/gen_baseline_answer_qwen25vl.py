"""Generate answers with Qwen2.5-VL model."""

import argparse
import json
import os
import sys
import time
import torch
import shortuuid
from tqdm import tqdm
from accelerate.utils import set_seed
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO

# 移除本地模型导入，直接使用transformers的标准加载方式
print("使用transformers的标准加载方式...")

# 固定随机种子（保证可复现）
set_seed(0)

# 第三方工具导入
from fastchat.llm_judge.common import load_questions

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

''' 
核心工作流程 ：
    1. 加载指定范围的问题集合（从question.jsonl）
    2. 根据总GPU数/单模型GPU数判断是否启用Ray分布式
    3. 将问题按chunk_size分块，分配到不同GPU/进程
    4. 调用get_model_answers()完成模型初始化+预热+生成答案
    5. 等待Ray分布式任务完成，收集所有结果
    6. 对答案文件去重+按question_id排序
'''
def run_eval(
        model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
    ):
    # 1. 加载问题（根据基准类型选择加载方式）
    if args.bench_name == "vqa":
        # 自定义VQA问题加载函数
        questions = []
        with open(question_file, "r") as fin:
            for line in fin:
                questions.append(json.loads(line))
        
        # 应用问题范围筛选
        if question_begin is not None:
            questions = questions[question_begin:]
        if question_end is not None:
            questions = questions[:question_end - (question_begin or 0)]
    else:
        # fastchat的标准加载函数
        questions = load_questions(question_file, question_begin, question_end)

    # 2. 是否启用Ray分布式：总GPU数 / 单模型GPU数 > 1 → 分布式
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    # 3. 包装生成函数：分布式场景用ray.remote包装，单机用原生函数
    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    # 4. 拆分问题：按chunk_size将问题分成多块，每块分配给一个GPU/进程
    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = [] # 存储Ray远程任务句柄（用于后续等待任务完成）
    for i in range(0, len(questions), chunk_size): # 按chunk_size遍历问题列表，分块调用生成函数
        ans_handles.append(
            # 每块问题调用一次生成函数
            get_answers_func(
                model_path,
                model_id,
                questions[i: i + chunk_size], # 本次进程处理的问题块
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )
    
    # 5. 等待分布式任务完成
    if use_ray:
        ray.get(ans_handles)


# 模型初始化 + 生成答案
# 适配Qwen2.5-VL的核心逻辑
@torch.inference_mode()
def get_model_answers(
        model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
    ):
    # 辅助函数：加载图像
    def load_image(image_path_or_url):
        if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)
        return image

    # 单轮对话生成，适用于warmup和正式推理
    def generate_single_turn(messages, processor, model, temperature):
        """
        Qwen2.5-VL适配的核心生成函数：处理单轮对话的prompt构造、生成、解码清理
        返回：生成的文本、新token数、idx（兼容原字段）、耗时
        """
        # 1、构造prompt并编码（适配Qwen的chat_template）
        # 对于多模态消息，我们需要特殊处理
        processed_messages = []
        for msg in messages:
            # 确保所有消息内容都转换为正确的多模态格式
            if isinstance(msg["content"], str):
                # 纯文本消息转换为列表格式
                content = [{"type": "text", "text": msg["content"]}]
            elif isinstance(msg["content"], list):
                # 已经是结构化内容，检查是否需要转换
                converted_content = []
                for item in msg["content"]:
                    if isinstance(item, str):
                        converted_content.append({"type": "text", "text": item})
                    else:
                        converted_content.append(item)
                content = converted_content
            else:
                # 未知格式，尝试转换为文本
                content = [{"type": "text", "text": str(msg["content"])}]
            
            processed_messages.append({
                "role": msg["role"],
                "content": content
            })
        
        # 使用processor处理消息
        inputs = processor.apply_chat_template(
            processed_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        # 2、模型生成（计时）
        torch.cuda.synchronize()
        start_time = time.time()
        # Qwen2.5-VL标准生成接口
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
        
        # 3、解码+清理特殊token
        # 只取生成的新token（去掉输入部分）
        output_ids = outputs[0][len(inputs.input_ids[0]):]
        # Qwen2.5-VL停止符处理
        stop_token_ids = [processor.tokenizer.eos_token_id]
        if stop_token_ids:
            stop_idx = [i for i, id in enumerate(output_ids) if id in stop_token_ids]
            if stop_idx:
                output_ids = output_ids[: stop_idx[0]]
        
        # 解码文本
        output = processor.batch_decode(
            [output_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        output = output.strip()
        
        # 兼容原返回格式（idx设为0，new_token为生成的token数）
        new_token = len(output_ids)
        idx = 0
        return output, new_token, idx, total_time

    # 初始化Qwen2.5-VL模型和Processor（核心适配）
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 使用transformers的标准方式加载模型
    # 尝试多种方式加载模型，优先使用专为条件生成设计的模型类
    model = None
    
    # 1. 首先尝试直接使用Qwen2_5_VLForConditionalGeneration（最适合的类）
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print("尝试使用Qwen2_5_VLForConditionalGeneration加载模型...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={i: max_gpu_memory for i in range(torch.cuda.device_count())}
        )
        print("✓ 使用Qwen2_5_VLForConditionalGeneration加载成功")
    except Exception as e:
        print(f"Qwen2_5_VLForConditionalGeneration加载失败: {e}")
    
    # 2. 如果失败，尝试使用AutoModelForCausalLM
    if model is None:
        try:
            from transformers import AutoModelForCausalLM
            print("尝试使用AutoModelForCausalLM加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={i: max_gpu_memory for i in range(torch.cuda.device_count())}
            )
            print("✓ 使用AutoModelForCausalLM加载成功")
        except Exception as e:
            print(f"AutoModelForCausalLM加载失败: {e}")
    
    # 3. 如果都失败，回退到AutoModel
    if model is None:
        try:
            from transformers import AutoModel
            print("尝试使用AutoModel加载模型...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={i: max_gpu_memory for i in range(torch.cuda.device_count())}
            )
            print("✓ 使用AutoModel加载成功")
        except Exception as e:
            print(f"AutoModel加载失败: {e}")
            raise RuntimeError(f"无法加载模型: {e}")
    
    # 验证模型结构
    print(f"\n模型类: {type(model).__name__}")
    print(f"模型是否有generate方法: {hasattr(model, 'generate')}")
    
    # 检查权重绑定情况（Qwen2.5-VL的特性）
    try:
        # 检查embed_tokens和lm_head是否存在且权重相等
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            
            # 检查各种可能的键名
            embed_keys = ['embed_tokens.weight', 'model.embed_tokens.weight']
            lm_head_keys = ['lm_head.weight', 'model.lm_head.weight']
            
            found_embed = None
            found_lm_head = None
            
            for key in embed_keys:
                if key in state_dict:
                    found_embed = key
                    break
            
            for key in lm_head_keys:
                if key in state_dict:
                    found_lm_head = key
                    break
            
            if found_embed and found_lm_head:
                print(f"找到embed_tokens权重: {found_embed}")
                print(f"找到lm_head权重: {found_lm_head}")
                
                # 检查权重是否相等
                weights_equal = torch.allclose(state_dict[found_embed], state_dict[found_lm_head])
                print(f"embed_tokens与lm_head权重是否相等: {weights_equal}")
                
                if weights_equal:
                    print("✓ 确认模型使用了权重绑定技术")
    except Exception as e:
        print(f"检查权重绑定时出错: {e}")
    
    # 检查并确保模型有生成能力
    if not hasattr(model, 'generate'):
        print("模型没有generate方法，但发现有can_generate属性，实现简单的生成功能...")
        
        # 如果有language_model，使用它
        if hasattr(model, 'language_model'):
            print("使用language_model作为基础模型...")
            base_model = model.language_model
        else:
            base_model = model
        
        print(f"基础模型类: {type(base_model).__name__}")
        print(f"基础模型属性: {[attr for attr in dir(base_model) if not attr.startswith('_')]}")
        
        # 创建一个简单的生成包装器
        def simple_generate(input_ids, attention_mask=None, max_new_tokens=100, temperature=0.0, **kwargs):
            """简单的生成实现，使用贪婪解码"""
            # 获取模型设备
            model_device = next(base_model.parameters()).device
            print(f"模型设备: {model_device}")
            
            # 确保输入张量在模型设备上
            generated_ids = input_ids.to(model_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)
            
            # 检查模型是否有lm_head
            has_lm_head = hasattr(base_model, 'lm_head')
            print(f"模型是否有lm_head: {has_lm_head}")
            
            for _ in range(max_new_tokens):
                # 前向传播
                outputs = base_model(generated_ids, attention_mask=attention_mask, **kwargs)
                
                # 获取最后一个token的隐藏状态
                last_hidden_state = outputs.last_hidden_state[:, -1, :]
                
                # 如果有lm_head，使用它生成logits
                if has_lm_head:
                    last_logits = base_model.lm_head(last_hidden_state)
                else:
                    # 如果没有lm_head，直接使用隐藏状态（不推荐，但作为备选）
                    print("警告：模型没有lm_head，直接使用last_hidden_state作为logits")
                    last_logits = last_hidden_state
                
                # 温度缩放
                if temperature > 0:
                    last_logits = last_logits / temperature
                
                # 贪婪解码
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                
                # 确保next_token在模型设备上
                next_token = next_token.to(model_device)
                
                # 添加到生成的序列中
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # 更新注意力掩码
                if attention_mask is not None:
                    new_mask = torch.ones((attention_mask.shape[0], 1), device=model_device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, new_mask], dim=1)
                
                # 检查是否生成了结束标记
                if next_token.item() == processor.tokenizer.eos_token_id:
                    break
            
            return generated_ids
        
        # 将生成方法添加到模型
        model.generate = simple_generate
        print("已添加simple_generate方法到模型")
    model.eval() # 模型设为评估模式

    # 模型预热：适配Qwen2.5-VL的预热逻辑
    if questions:
        warmup_question = questions[0]
        for _ in range(3):
            torch.manual_seed(0)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            for j in range(len(warmup_question["turns"])):
                # 检查问题是否包含图像
                if "image" in warmup_question and j == 0:
                    # 构造多模态消息
                    content = [
                        {"type": "image", "image": load_image(warmup_question["image"])},
                        {"type": "text", "text": warmup_question["turns"][j]}
                    ]
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": warmup_question["turns"][j]})
                
                # 模拟生成过程
                output, _, _, _ = generate_single_turn(
                    messages, 
                    processor, 
                    model, 
                    temperature
                )
                messages.append({"role": "assistant", "content": output})
        print('Warmup done')

    # 正式批量生成答案
    for question in tqdm(questions, desc="Generating answers"):
        choices = []
        # 每个问题生成num_choices个答案
        for i in range(num_choices):
            torch.manual_seed(i)
            # 系统提示（固定）
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            # 存储多轮对话的回复
            turns, idxs, new_tokens, wall_time = [], [], [], []
            
            # 多轮对话处理（适配Qwen格式）
            for j in range(len(question["turns"])):
                # 构造本轮用户问题
                if "image" in question and j == 0:
                    # 如果问题包含图像，构造多模态消息
                    content = [
                        {"type": "image", "image": load_image(question["image"])},
                        {"type": "text", "text": question["turns"][j]}
                    ]
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": question["turns"][j]})
                
                # 调用Qwen适配的生成函数
                output, new_token, idx, total_time = generate_single_turn(
                    messages, processor, model, temperature
                )

                # 记录本轮结果
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                # 将本轮回复加入对话历史
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # 收集当前问题的所有候选答案
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # 写入答案文件
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


''' 
去重：如果同一个问题生成了多次答案，保留最后一次（字典覆盖）；
排序：按question_id升序排列，方便后续评估 / 查看
'''
def reorg_answer_file(answer_file):
    # 对答案文件进行排序和去重
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            answers[data["question_id"]] = line # 去重：按question_id保留最后一个

    # 按question_id排序
    with open(answer_file, "w") as fout:
        for qid in sorted(list(answers.keys())):
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Qwen2.5-VL模型路径配置（简化为单路径）
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/Models/Qwen2.5-VL-7B-Instruct/",
        help="Path to Qwen2.5-VL model"
    )
    # 模型基础配置
    parser.add_argument("--model-id", type=str, default="qwen2.5vl-instruct")
    # 测试问题配置
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-file", type=str, help="The input question file.")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    # 生成参数配置
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    # 分布式配置
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default="80GiB")

    args = parser.parse_args()

    # model_id 追加温度
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    
    # 确定问题文件和答案文件路径
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    # 优先使用--question-file参数
    if args.question_file:
        question_file = args.question_file
    else:
        # 如果没有提供问题文件参数，尝试常见的默认路径
        default_paths = [
            f"{parent_dir}/data/{args.bench_name}/question.jsonl",
            f"/workspace/datasets/{args.bench_name}/vqa_questions.jsonl",  # 用户的实际数据路径
            f"/workspace/datasets/{args.bench_name}/{args.bench_name}_questions.jsonl"
        ]
        
        found = False
        for path in default_paths:
            if os.path.exists(path):
                question_file = path
                found = True
                print(f"Found question file at: {question_file}")
                break
        
        if not found:
            print(f"Error: Could not find question file.")
            print(f"Please provide a valid path using --question-file parameter.")
            print(f"Tried paths:")
            for path in default_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # 答案文件路径
    answer_file = args.answer_file or f"{args.bench_name}/{args.model_id}.jsonl"
    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
    print("Answer generation completed and reorganized!")
