"""Generate answers with Eagle3 + Llama3
核心功能：基于Eagle3加速的Llama3模型批量生成问答答案，支持分布式推理
"""
import argparse
import json
import os
import time
import torch
import shortuuid
from tqdm import tqdm
from accelerate.utils import set_seed

# 第三方工具导入
from fastchat.llm_judge.common import load_questions

# Eagle3核心模块（仅保留Eagle3相关）
from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor

# ===================== 1. 基础配置 =====================
# 固定随机种子（保证可复现）
set_seed(0)

# 系统提示词
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

# ===================== 2. 分布式任务协调 =====================
def run_eval(
        base_model_path,
        ea_model_path,
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
    # 1. 加载指定范围的问题
    questions = load_questions(question_file, question_begin, question_end)
    print(f"Loaded {len(questions)} questions (range: {question_begin}-{question_end})")

    # 2. 判断是否启用Ray分布式（总GPU数/单模型GPU数 > 1）
    assert num_gpus_total % num_gpus_per_model == 0, "总GPU数需能被单模型GPU数整除"
    use_ray = num_gpus_total // num_gpus_per_model > 1

    # 3. 包装生成函数（分布式/单机）
    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    # 4. 问题分块 + 提交任务
    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
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
        ray.shutdown()


# ===================== 3. 模型推理核心 =====================
@torch.inference_mode()
def get_model_answers(
    base_model_path,
    ea_model_path,
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
    """
    核心逻辑：加载Eagle3模型 + 预热 + 批量生成答案
    """
    # ---------------------- 3.1 工具函数：单轮生成 ----------------------
    def generate_single_turn(messages, tokenizer, model, temperature):
        """
        单轮对话生成：构造prompt → 模型推理 → 解码清理
        返回：生成文本、新token数、idx、耗时
        """
        # 1. 构造Llama3 Chat模板（适配多轮对话格式）
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 2. 编码prompt为input_ids（不添加特殊token，适配Llama3）
        input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
        # 3. 模型推理（计时）
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
            log=True,
            is_llama3=True,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        # 4. 处理输出：截断停止标记 + 解码 + 清理特殊token
        output_ids = output_ids[0][len(input_ids[0]):]  # 移除输入部分，只保留生成内容
        # 停止标记（Llama3的EOS和EOT）
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        if stop_token_ids:
            stop_idx = [i for i, id in enumerate(output_ids) if id in stop_token_ids]
            if stop_idx:
                output_ids = output_ids[:stop_idx[0]]
        # 解码 + 清理特殊token
        output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        for special_tok in tokenizer.special_tokens_map.values():
            if isinstance(special_tok, list):
                for tok in special_tok:
                    output = output.replace(tok, "")
            else:
                output = output.replace(special_tok, "")
        output = output.strip()
        return output, new_token, idx, total_time

    # ---------------------- 3.2 加载Eagle3模型 ----------------------
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    model.eval()  # 评估模式（禁用Dropout）
    print(f"Model loaded successfully (eval mode: {not model.training})")

    # ---------------------- 3.3 模型预热（避免首次推理卡顿） ----------------------
    if questions:
        warmup_question = questions[0]
        for _ in range(3):  # 预热3轮
            torch.manual_seed(0)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for j in range(len(warmup_question["turns"])):
                messages.append({"role": "user", "content": warmup_question["turns"][j]})
                # 模拟生成
                output, _, _, _ = generate_single_turn(messages, tokenizer, model, temperature)
                messages.append({"role": "assistant", "content": output})
    print("Warmup completed")

    # ---------------------- 3.4 批量生成答案 ----------------------
    for question in tqdm(questions, desc="Generating answers"):
        choices = []
        # 每个问题生成num_choices个答案（默认1）
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            turns, idxs, new_tokens, wall_time = [], [], [], []
            # 多轮对话处理
            for j in range(len(question["turns"])):
                messages.append({"role": "user", "content": question["turns"][j]})
                # 单轮生成
                output, new_token, idx, total_time = generate_single_turn(
                    messages, tokenizer, model, temperature
                )
                # 记录结果
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                # 追加助手回复到对话历史
                messages.append({"role": "assistant", "content": output})
            # 收集当前候选答案
            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time
            })
        # ---------------------- 3.5 写入答案文件 ----------------------
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

# ===================== 4. 答案文件整理（去重+排序） =====================
def reorg_answer_file(answer_file):
    """按question_id排序并去重（保留最后一个答案）"""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            answers[data["question_id"]] = line  # 去重：覆盖重复的question_id
    # 按question_id升序写入
    with open(answer_file, "w") as fout:
        for qid in sorted(answers.keys()):
            fout.write(answers[qid])


# ===================== 5. 主函数（参数解析+启动流程） =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eagle3 + Llama3 生成")
    # 模型路径配置
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/workspace/Models/EAGLE3-LLaMA3.1-Instruct-8B/",
    )
    parser.add_argument(
        "--base-model-path", 
        type=str, 
        default="/workspace/Models/Llama-3.1-8B-Instruct/",
    )
    # 模型核心参数（Eagle3专属）
    parser.add_argument("--total-token", type=int, default=60,
                        help="total-token = 树中草拟令牌总数 + 1")
    parser.add_argument("--depth", type=int, default=5,
                        help="depth = 最大草拟长度 - 1")
    parser.add_argument("--top-k", type=int, default=10,
                        help="每层草拟令牌的最大数量")
    # 推理配置
    parser.add_argument("--model-id", type=str, default="llama3_8b40", help="模型标识")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（0为贪心）")
    parser.add_argument("--num-choices", type=int, default=1, help="每个问题生成的答案数")
    parser.add_argument("--max-new-token", type=int, default=1024, help="最大生成token数")
    # 分布式配置
    parser.add_argument("--num-gpus-per-model", type=int, default=1, help="单模型占用GPU数")
    parser.add_argument("--num-gpus-total", type=int, default=1, help="总GPU数")
    parser.add_argument("--max-gpu-memory", type=str, help="单GPU最大显存占用")
    # 数据配置
    parser.add_argument("--bench-name", type=str, default="mt_bench", help="评测集名称")
    parser.add_argument("--question-begin", type=int, help="问题起始索引")
    parser.add_argument("--question-end", type=int, help="问题结束索引")
    parser.add_argument("--answer-file", type=str, help="答案输出文件路径（默认：results/[bench-name]/[model-id].jsonl）")
    
    args = parser.parse_args()

    # 补充模型ID（追加温度参数）
    args.model_id = f"{args.model_id}-temperature-{args.temperature}"

    # 确定问题文件和答案文件路径
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    # 答案文件默认路径（优先级：命令行参数 > 自动生成）
    if not args.answer_file:
        project_root = os.path.dirname(parent_dir)
        args.answer_file = f"{project_root}/results/{args.bench_name}/{args.model_id}.jsonl"
    print(f"[Config] Question file: {question_file}")
    print(f"[Config] Answer file: {args.answer_file}")

    # 启动核心流程
    run_eval(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=args.answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
        args=args
    )

    # 整理答案文件
    reorg_answer_file(args.answer_file)
    print(f"✅ All done! Answer file: {args.answer_file}")
