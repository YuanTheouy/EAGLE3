'''

1.  参数配置模块（命令行+DeepSpeed+训练超参）
2.  数据处理模块（数据集构建+预处理+数据加载）
3.  模型与优化器初始化模块（EAGLE 模型+AdamW+DeepSpeed 封装）
4.  分布式训练环境配置模块（rank 分配+WandB 日志+采样器）
5.  断点续训模块（检查点查找与加载）
6.  核心训练&验证模块（epoch 循环+前向传播+反向传播+指标统计+保存）

'''

'''
1. 参数配置部分
    1.1 命令行参数（argparse）
    1.2 DeepSpeed 配置加载
    1.3 训练超参配置（train_config 字典）
'''

import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/workspace/Models/Llama-3.1-8B-Instruct')
parser.add_argument('--trainpath', type=str,
                    default="/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_train_T00.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_test_T00.jsonl")
parser.add_argument('--savedir', type=str, default='/workspace/Models/EAGLE-LLama-3.1-v3')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
# 为命令行参数添加 DeepSpeed 配置项（如 --deepspeed 指定配置文件路径
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 40,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    # "gradient_checkpointing": True
    "gradient_checkpointing": False
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup



'''
2. 数据处理模块：构建适配 EAGLE 训练的数据集与数据加载器
2.1 数据集构建函数 build_dataset_rank
2.2 数据拼接器 DataCollatorWithPadding
2.3 数据加载器（DataLoader）与分布式采样器（DistributedSampler）
'''

def build_dataset_rank(
        tokenizer, datapath
):
    # 读取数据文件，加载为 JSON 格式的数据集，打乱训练集
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    # 后续数据操作基于此副本，保留原始数据集
    ds1 = ds
    # 记录原始数据集列名，比如 id conversations，用于后续删除原始列
    original_columns1 = ds1.column_names
    # 指定后续map操作的并行进程数
    num_proc = 8

    def preprocess_function(examples):
        '''
        1 初始化输出字典
        '''
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            '''
            2 对话消息格式化：构建标准 Chat 格式
            '''
            # 初始化系统提示词，系统行为不变
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            # 角色映射与对话轮次校验
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
            source = examples['conversations'][i] # 从原始数据中提取第i个样本的conversations字段
            # 过滤无效对话和非用户开头对话
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            # 遍历对话，拼接至 message 列表
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                # 校验角色是否匹配当前轮次
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                # 拼接每一轮对话
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            # 应用Llama-3.1的chat模板，生成对话字符串
            '''
            这部分返回的是字符串（因为tokenize=False），比如会生成这样的内容：<|start_header_id|>system<|end_header_id|> You are a helpful assistant...<|eot_id|><|start_header_id|>user<|end_header_id|> 你好<|eot_id|><|start_header_id|>assistant<|end_header_id|> 您好，有什么可以帮您的？<|eot_id|>

            只是训练，不需要生成提示，比如 LLaMA 的 <|start_header_id|>assistant<|end_header_id|>  这样的结尾提示是不需要的
            那这个messages本来是长什么样的呢？

            '''
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False, # 不添加生成提示，因为我们是训练对话模型
            )

            '''
            3 分词长度过滤
            '''
            # 兜底：如果tokenizer没有pad_token_id，设置为unk_token_id
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            # 对格式化后的对话字符串进行 Tokenize，生成输入ID序列
            '''
            return_tensors="pt", 默认返回列表，该参数让返回值为 torch.Tensor
            add_special_tokens=False：因为 apply_chat_template 已经生成了 LLaMA 所需的特殊标记（<|eot_id|> 等），这里如果再添加，会重复出现 <s>/</s> 等标记，导致模型报错；
            .input_ids[0]：tokenizer 返回的 input_ids 默认是2D 张量（形状：[1, 序列长度]，批量维度为 1），[0] 是为了去除批量维度，得到 1D 张量（形状：[序列长度]），方便后续样本处理。
            '''
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            # 过滤超长文本，防止溢出
            # filtering out the samples which is longer than max_len
            if len(input_ids) > train_config["max_len"]:
                continue

            '''
            4 损失掩码构建
            '''
            # 初始化损失掩码，全1向量，长度与输入ID序列一致
            loss_mask = torch.ones_like(input_ids)
            # print(i)
            # 定义 LLaMA 对话特特殊分隔符，用于分割用户轮次和助手轮次
            # 助手轮次分隔符，用于将助手回复与用户输入隔开
            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            # 用户轮次分隔符，用于将用户输入与助手回复隔开
            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            
            # 分割对话轮次，提取有效训练片段
            # 按用户轮次分割对话
            turns = conversation.split(sep2)
            # 还原首轮用户轮次，包含用户输入和助手回复
            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            # 初始化当前长度指针
            cur_len = 1
            loss_mask[:cur_len] = 0 #将开头token置零，忽略开头特殊标记
            
            
            # 遍历每一轮对话，对用户输入部分进行掩码
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                
                # 按照助手分隔符拆分当前轮次，提取 用户输入 和 助手回复
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                # 计算用户输入部分的长度 硬编码 -1 适配了 Llama3 tokenizer 的特殊 token 偏移
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions# 
                # 将用户轮次的loss_mask置0，忽略用户输入部分
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                # 更新当前长度指针，适配多轮对话偏移
                cur_len += turn_len
                if i != 0:
                    cur_len += 3
                # cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1
            
            # 将当前轮次之后的所有token都置零 
            loss_mask[cur_len:] = 0
            # 全1张量。表示所有token都参与注意力的计算
            attention_mask = torch.ones_like(loss_mask)

            # 收集与处理结果
            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    '''
    对数据集 ds1 进行批量处理，应用 preprocess_function 函数，生成新的特征列（input_ids, loss_mask, attention_mask）
    1. 批量处理：batched=True 表示对数据集进行批量处理，num_proc=num_proc 表示使用多个进程并行处理
    2. 列删除：remove_columns=original_columns1 表示在处理完成后删除原始列，保留新生成的列
    3. 缓存禁用：load_from_cache_file=False 表示不使用缓存文件，每次都重新处理数据
    '''
    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    # 将当前批次的张量padding到统一长度，在第一位度填充0，将子序列补至N [批次大小，子序列数，序列长度]
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors
    
    # 将当前批次的2D张量padding到统一长度，在第一维度度填充0，将子序列补至N [批次大小，序列长度]
    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    # 对当前批次的特征进行padding，返回填充后的张量
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 计算当前批次的最大序列长度
        max_length = max(item['input_ids'].shape[1] for item in features)
        # 对input_ids、attention_mask、loss_mask进行padding
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        # 构建批次字典并返回。
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

# 确保保存目录存在
os.makedirs(args.savedir, exist_ok=True)


if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])


'''
模型与优化器初始化模块：构建 EAGLE 模型并封装 DeepSpeed
'''

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)

# 定义原生 AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5),
                                                     model_parameters=model.parameters(),
                                                     )

deepspeed.comm.barrier()


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)



model.scandata(args.trainpath, args.basepath)


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]



global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    import wandb

    # wandb.login(key="54225ff98513185d3eb3c41c709b1f8a65a06dee")
    # wandb.init(project="lamma3-8b-v1", entity="1192445377", config=ds_config)
    
    # 使用环境变量获取API Key（可选）
    api_key = os.environ.get("WANDB_API_KEY", "")
    wandb.login(key=api_key)
    
    # 配置保存方式
    wandb_mode = os.environ.get("WANDB_MODE", "online")  # online/offline
    wandb_dir = os.environ.get("WANDB_DIR", "./wandb")  # 自定义本地目录
    
    wandb.init(
        project="lamma3-8b-v1",
        entity="1192445377",
        config=ds_config,
        mode=wandb_mode,
        dir=wandb_dir
    )

os.makedirs(args.savedir, exist_ok=True)

sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)



for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]


    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()

        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)


        model_engine.step()

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochacc_{i}": acc_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochploss_{i}": loss_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(rank),
                                                   loss_mask=data["loss_mask"],
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochacc_{i}": acc_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochploss_{i}": loss_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")
    # clear out the redundance cahce after each step
    torch.cuda.empty_cache()

    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")