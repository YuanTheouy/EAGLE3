import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/workspace/Models/Llama-3.1-8B-Instruct')
parser.add_argument('--trainpath', type=str,
                    default="/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_train_T00.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_test_T00.jsonl")
parser.add_argument('--savedir', type=str, default='/workspace/Models/EAGLE-LLama-3.1-v3')
parser.add_argument('--text_model_path', type=str, default=None, help="Path to pre-trained text model checkpoint (Stage 1)")
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": args.num_epochs,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": args.basepath + "/config.json",
    # "gradient_checkpointing": True
    "gradient_checkpointing": False
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets_vl import Model
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



from PIL import Image

def build_dataset_rank(
        tokenizer, datapath
):

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": [],
            "pixel_values": [],
            "image_grid_thw": []
        }
        for i in range(len(examples['id'])):
            # Load conversation
            conv = examples['conversations'][i]
            # Helper to load image
            def load_image(image_path):
                return Image.open(image_path).convert("RGB")

            qwen_messages = []
            
            # Handle image
            # Note: Dataset format check
            # User format: {"id": "...", "conversations": [{"from": "human", "value": "...", "image": "path"}]}
            # Script format assumption: "image" column or inside conversation
            
            image_path = None
            if 'image' in examples:
                image_path = examples['image'][i]
            elif 'conversations' in examples:
                 # Check inside the first turn of conversation
                 if len(examples['conversations'][i]) > 0 and 'image' in examples['conversations'][i][0]:
                     image_path = examples['conversations'][i][0]['image']
            
            image_obj = None
            if image_path:
                try:
                    image_obj = load_image(image_path)
                except Exception as e:
                    print(f"Failed to load image: {image_path}, error: {e}")
                    continue
            
            for idx, turn in enumerate(conv):
                role = "user" if turn['from'] == "human" else "assistant"
                content = turn['value']
                
                content_list = []
                # If first user turn and we have an image
                if role == "user" and idx == 0 and image_obj:
                    content_list.append({"type": "image", "image": image_obj})
                
                content_list.append({"type": "text", "text": content})
                qwen_messages.append({"role": role, "content": content_list})

            # Prepare inputs
            text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=False)
            
            # Process with images
            image_inputs = [image_obj] if image_obj else None
            
            # Processor call
            try:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt"
                )
            except Exception as e:
                 print(f"Processor error: {e}")
                 continue
            
            input_ids = inputs.input_ids[0]
            # Pixel values and grid
            pixel_values = inputs.pixel_values if hasattr(inputs, "pixel_values") else None
            image_grid_thw = inputs.image_grid_thw if hasattr(inputs, "image_grid_thw") else None
            
            if len(input_ids) > train_config["max_len"]:
                continue

            # Loss Mask
            loss_mask = torch.zeros_like(input_ids)
            
            # Find assistant segments
            # IDs for markers (dynamic lookup)
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            # assistant token might be split or whole depending on tokenizer
            # For Qwen2.5, "assistant" is usually one token.
            assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
            
            # Scan
            j = 0
            while j < len(input_ids):
                # Look for <|im_start|> assistant
                if input_ids[j] == im_start_id:
                    # Check if next tokens match assistant
                    match = True
                    for k, aid in enumerate(assistant_ids):
                        if j + 1 + k >= len(input_ids) or input_ids[j + 1 + k] != aid:
                            match = False
                            break
                    
                    if match:
                         # Found header. Move past it.
                         start = j + 1 + len(assistant_ids) 
                         # Skip newline if present (optional, usually Qwen template adds \n)
                         # We can just start masking from here.
                         
                         # Find end
                         end = start
                         while end < len(input_ids) and input_ids[end] != im_end_id:
                             end += 1
                         
                         # Mask this range (start to end)
                         loss_mask[start:end] = 1
                         j = end
                    else:
                        j += 1
                else:
                    j += 1
            
            # Add to list
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(inputs.attention_mask[0][None, :])
            
            if pixel_values is not None:
                new_examples["pixel_values"].append(pixel_values)
            else:
                 new_examples["pixel_values"].append(None)
            
            if image_grid_thw is not None:
                new_examples["image_grid_thw"].append(image_grid_thw)
            else:
                new_examples["image_grid_thw"].append(None)

        return new_examples

    # 动态数据处理适配器（避免 map 产生巨大的缓存文件）
    def transform_adapter(examples):
        # 检查是否为单样本（通过检查 'id' 是否为 list）
        if 'id' in examples and not isinstance(examples['id'], list):
             # 单样本 -> Batch (列表化)
             examples_batch = {k: [v] for k, v in examples.items()}
             # 调用原处理函数
             result_batch = preprocess_function(examples_batch)
             # Batch -> 单样本 (取第一个元素)
             # 注意：preprocess_function 返回的字典中，每个 value 都是 list
             return {k: v[0] if v is not None else None for k, v in result_batch.items()}
        else:
             # 已经是 Batch（如果 DataLoader 启用了 batch_sampler）
             return preprocess_function(examples)

    # 使用 set_transform 进行在线处理，替代 map 的离线缓存机制
    # 这解决了图文数据集处理后占用数 TB 磁盘空间的问题
    ds1.set_transform(transform_adapter)
    
    # ds1 = ds1.map(
    #    preprocess_function,
    #    batched=True,
    #    num_proc=num_proc,
    #    remove_columns=original_columns1,
    #    load_from_cache_file=False
    # )
    
    # Remove None values if any
    # Or handle in collate
    # ds1.set_format(type="torch") # set_transform 已经接管了输出格式，set_format 可能不再需要或冲突
    return ds1


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        # Vision inputs
        batch_pixel_values = None
        batch_image_grid_thw = None
        
        pixel_values_list = [item['pixel_values'] for item in features if item['pixel_values'] is not None]
        image_grid_thw_list = [item['image_grid_thw'] for item in features if item['image_grid_thw'] is not None]
        
        # Check if they are not empty/None
        # Note: map might convert None to something else or keep it.
        # We need to filter valid tensors.
        
        valid_pixel_values = []
        valid_image_grid = []
        
        for pv in pixel_values_list:
            if isinstance(pv, torch.Tensor) and pv.numel() > 0:
                valid_pixel_values.append(pv)
        
        for ig in image_grid_thw_list:
            if isinstance(ig, torch.Tensor) and ig.numel() > 0:
                valid_image_grid.append(ig)
                
        if valid_pixel_values:
            batch_pixel_values = torch.cat(valid_pixel_values, dim=0)
            batch_image_grid_thw = torch.cat(valid_image_grid, dim=0)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw
        }
        return batch

# ==================== 【插入这段暴力修正代码】 ====================
import os

# 1. 打印调试信息，看看到底发生了什么
print(f"DEBUG: Original args.local_rank = {args.local_rank}")
print(f"DEBUG: os.environ['LOCAL_RANK'] = {os.environ.get('LOCAL_RANK', 'Not Set')}")

# 2. 暴力覆盖：如果 args 是 -1，但环境变量里有值，强制覆盖！
if args.local_rank == -1:
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"✅ FIXED: Forced args.local_rank to {args.local_rank} from Env Var")
    else:
        # 如果环境变量也没有，手动根据 CUDA_VISIBLE_DEVICES 猜一个（最后的兜底）
        # 注意：这通常不需要，Launcher 100% 会给环境变量
        print("❌ CRITICAL: LOCAL_RANK not found in Env!")

# 3. 再次确认
print(f"DEBUG: Final args.local_rank used for init = {args.local_rank}")
# =================================================================

os.makedirs(args.savedir, exist_ok=True)


if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

deepspeed.init_distributed(dist_backend="nccl")

# config = EConfig.from_pretrained(train_config["config_path"])
config = EConfig.from_pretrained(args.basepath)
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)

# 定义原生 AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


# model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
#                                                      model=model,
#                                                      optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5),
#                                                      model_parameters=model.parameters(),
#                                                      )

# deepspeed.comm.barrier()


# tokenizer = AutoTokenizer.from_pretrained(args.basepath)
processor = AutoProcessor.from_pretrained(args.basepath, trust_remote_code=True)
tokenizer = processor.tokenizer

traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)



criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]


model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5),
                                                     model_parameters=model.parameters(),
                                                     )

# Ensure scandata runs on all ranks for ZeRO-3 gathering, but file write is guarded in scandata
if hasattr(model_engine, "module"):
    model_engine.module.scandata(args.trainpath, args.basepath)
else:
    model_engine.scandata(args.trainpath, args.basepath)

deepspeed.comm.barrier()


global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    import wandb

    
    # 使用环境变量获取API Key（可选）
    api_key = os.environ.get("WANDB_API_KEY", "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")
    wandb.login(key=api_key)
    
    # 配置保存方式
    wandb_mode = os.environ.get("WANDB_MODE", "online")  # online/offline
    wandb_dir = os.environ.get("WANDB_DIR", "./wandb")  # 自定义本地目录
    
    wandb.init(
        project="qwen25vl",
        entity="1192445377-zhejiang-university",
        config=ds_config,
        mode=wandb_mode,
        dir=wandb_dir
    )

os.makedirs(args.savedir, exist_ok=True)

sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, 
                         num_workers=train_config["num_workers"], pin_memory=True, prefetch_factor=4,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, 
                          num_workers=train_config["num_workers"], pin_memory=True, prefetch_factor=4,
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
elif args.text_model_path:
    print(f"load stage 1 model from {args.text_model_path}")
    if os.path.isdir(args.text_model_path):
        # Assume DeepSpeed checkpoint
        model_engine.load_checkpoint(args.text_model_path)
    else:
        # Assume state_dict file (e.g., .pt or .bin)
        state_dict = torch.load(args.text_model_path, map_location="cpu")
        # Handle key mismatch if necessary (e.g. "module." prefix)
        # DeepSpeed engine wraps model in .module
        model_engine.module.load_state_dict(state_dict, strict=False)



for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]


    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()
        
        # Handle vision inputs
        pixel_values = data.get("pixel_values")
        image_grid_thw = data.get("image_grid_thw")
        if pixel_values is not None:
             pixel_values = pixel_values.to(rank).to(torch.float16) 
        if image_grid_thw is not None:
             image_grid_thw = image_grid_thw.to(rank)

        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               pixel_values=pixel_values,
                                               image_grid_thw=image_grid_thw
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
            # Handle vision inputs
            pixel_values = data.get("pixel_values")
            image_grid_thw = data.get("image_grid_thw")
            if pixel_values is not None:
                 pixel_values = pixel_values.to(rank).to(torch.float16) 
            if image_grid_thw is not None:
                 image_grid_thw = image_grid_thw.to(rank)
                 
            plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(rank),
                                                   loss_mask=data["loss_mask"],
                                                   pixel_values=pixel_values,
                                                   image_grid_thw=image_grid_thw
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