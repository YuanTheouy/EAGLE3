import copy
import random
import time
from typing import Any, List, Tuple

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# 核心常量（仅保留必要的）
TOPK = 10  # sparse tree topk
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Timer:
    """简易计时上下文管理器，同步CUDA后计算耗时"""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f"[{Timer.__name__}] {self.name} took {elapsed:.4f} seconds")


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature <1e-5 or temperature == 1.0:
        return None
    processor_list.append(TemperatureLogitsWarper(temperature)) # 调整生成的随机性，本质是对logits除以温度值的缩放后再做softmax
    if repetition_penalty > 1.0: # 重复惩罚处理器，抑制文本生成重复内容
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty)) 
    if 1e-8 <= top_p < 1.0: # 和TopK采样类似，通过累计概率筛选出前top_p的token
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0: # 限制生成的token数量，只考虑top_k个最可能的token
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """将路径填充到指定长度"""
    if len(path) >= length:
        return path.copy()
    return path + [pad_value] * (length - len(path))

def generate_tree_buffers(tree_choices:List[List[int]], device="cuda"):
    """排序路径→统计深度→构建 4 类核心张量→整理输出"""
    def _custom_sort_key(lst: List[int],maxitem:int) -> List[int]:
        """自定义排序key生成函数"""
        return [x if x>=0 else maxitem for x in lst]
    
    with Timer("tree_choice_sort"):
        """阶段一：排序token路径，确保短路径优先"""
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x),x)) # 对路径进行自定义排序，确保短路径优先
        tree_len = len(sorted_tree_choices) + 1

        # 统计各深度的路径数量
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth: # 统计各深度的路径数量
                depth_counts.append(0)
                prev_depth = depth
            depth_counts[depth - 1] += 1

        # 构建树注意力mask
        tree_attn_mask  = torch.eye(tree_len, tree_len) # torch.eye 的意思是生成一个对角线为1的矩阵，其他位置为0
        tree_attn_mask[:, 0] = 1 # root 节点对所有节点可见，即第0行全为1
        start_idx = 0
        for depth,count in enumerate[Any](depth_counts):
            for j in range(count):
                path = sorted_tree_choices[start_idx + j]
                if len(path) == 1:
                    continue
                # 找到祖先路径的索引
                ancestor_idxs = [
                    sorted_tree_choices.index(path[:c+1])+1 # +1 for root node offset
                    for c in range(len(path)-1)
                ]
                tree_attn_mask[start_idx + j + 1, ancestor_idxs] = 1
            start_idx += count

        # 2. 构建树索引、父索引、偏置索引
        tree_indices = torch.zeros(tree_len, dtype=torch.long, device=device)
        p_indices = [-1] + [0] * (tree_len - 1)  # -1 for root
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0  # root索引
        
        start_idx = 0
        global_bias = 0
        for depth, count in enumerate(depth_counts):
            inlayer_bias = 0
            current_b = []
            parent_path = None
            for j in range(count):
                path = sorted_tree_choices[start_idx + j]
                cur_parent = path[:-1]

                # 更新偏置
                if j != 0 and cur_parent != parent_path:
                    global_bias += 1
                    inlayer_bias += 1
                    parent_path = cur_parent
                    current_b = []
                elif j == 0:
                    parent_path = cur_parent

                # 计算树索引
                tree_idx_val = path[-1] + TOPK * (depth + 1 + global_bias) + 1
                tree_indices[start_idx + j + 1] = tree_idx_val
                p_indices[start_idx + j + 1] = inlayer_bias
                b_indices[start_idx + j] = copy.deepcopy(current_b)
                current_b.append(tree_idx_val)
            start_idx += count

        # 3. 构建树位置ID
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long, device=device)
        start_idx = 0
        for depth, count in enumerate(depth_counts):
            tree_position_ids[start_idx + 1: start_idx + count + 1] = depth + 1
            start_idx += count

        # 4. 构建检索索引
        retrieve_paths = []
        retrieve_indices_nest = []
        for path in reversed(sorted_tree_choices):
            if path in retrieve_paths:
                continue
            retrieve_idx = [
                sorted_tree_choices.index(path[:c+1]) + 1  # +1 for root offset
                for c in range(len(path))
            ]
            retrieve_indices_nest.append(retrieve_idx)
            retrieve_paths.extend([path[:c+1] for c in range(len(path))])

        # 填充检索索引到最大长度
        max_retrieve_len = max(len(x) for x in retrieve_indices_nest) if retrieve_indices_nest else 0
        retrieve_indices = [pad_path(p, max_retrieve_len) for p in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=device)
        # 拼接root索引（0）在最前面
        retrieve_indices = torch.cat([
            torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long, device=device),
            retrieve_indices
        ], dim=1)

        # 自定义排序检索索引
        maxitem = retrieve_indices.max().item() + 5 if retrieve_indices.numel() > 0 else 0
        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=lambda x: _custom_sort_key(x, maxitem))
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=device)

    # 整理输出缓冲区
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),  # [1,1,tree_len,tree_len]
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    return tree_buffers


def initialize_tree(
    input_ids: torch.Tensor,
    model,
    past_key_values,
    logits_processor: LogitsProcessorList = None
    ):
    """初始化Eagle3树结构，生成初始draft tokens和树参数"""
    # 前向传播获取基础输出
    outputs, orig_logits, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    # 采样第一个token
    if logits_processor is not None:
        logits = orig_logits[:, -1]
        logits = logits_processor(None, logits)
        probs = torch.nn.functional.softmax(logits, dim=1)
        sample_token = torch.multinomial(probs, 1)
    else:
        sample_token = torch.argmax(orig_logits[:, -1])  # 去掉dim=-1，返回标量
        sample_token = sample_token[None, None]  # 扩展为[1,1]（2维）

    # 更新输入ID（拼接采样的token）
    input_ids = torch.cat([input_ids, sample_token.to(input_ids.device)], dim=1)

    # Eagle3专属逻辑：拼接最后3层hidden states
    ea_device = model.ea_layer.lm_head.weight.device
    # 确保hidden states在同一设备
    hidden_states = [x.to(ea_device) for x in outputs["hidden_states"]]
    hidden_states = torch.cat(hidden_states[-3:], dim=-1)

    # 生成树结构的draft tokens和索引
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        hidden_states, input_ids, model.base_model.lm_head, logits_processor
    )

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig_logits, hidden_states, sample_token


def reset_tree_mode(model):
    """重置模型的树模式相关参数"""
    # [MODIFIED] Check both base_model.model and language_model for Qwen2.5-VL compatibility
    target_models = [model.base_model.model]
    if hasattr(model.base_model.model, "language_model"):
        target_models.append(model.base_model.model.language_model)
    
    for m in target_models:
        if hasattr(m, "tree_mask"):
            m.tree_mask = None
        if hasattr(m, "tree_mode"):
            m.tree_mode = None

def reset_past_key_values(passed_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """重置Past Key Values的current_length为0（仅支持batch_size=1）"""
    if not passed_key_values:
        return passed_key_values
    for layer_idx in range(len(passed_key_values)):
        for kv_idx in range(2):  # 0=key, 1=value
            if hasattr(passed_key_values[layer_idx][kv_idx], "current_length"):
                passed_key_values[layer_idx][kv_idx].current_length.fill_(0)
    return passed_key_values


def generate_candidates(
    tree_logits: torch.Tensor,
    tree_indices: torch.Tensor,
    retrieve_indices: torch.Tensor,
    sample_token: torch.Tensor,
    logits_processor: LogitsProcessorList = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """基于树logits生成候选token序列"""
    # 设备对齐
    sample_token = sample_token.to(tree_indices.device)
    
    # 拼接采样token和树logits
    candidate_logit = sample_token[0]
    candidate_tree_logits = tree_logits.view(-1)
    candidates = torch.cat([candidate_logit, candidate_tree_logits], dim=-1)

    # 按树索引筛选候选
    tree_candidates = candidates[tree_indices]
    # 扩展候选（补-1）
    tree_candidates_ext = torch.cat([
        tree_candidates,
        torch.tensor([-1], dtype=torch.long, device=tree_candidates.device)
    ], dim=0)
    # 按检索索引生成笛卡尔候选
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # 增加batch维度（适配模型输入）
    tree_candidates = tree_candidates.unsqueeze(0)

    return cart_candidates, tree_candidates


def tree_decoding(
    model,
    tree_candidates: torch.Tensor,
    past_key_values,
    tree_position_ids: torch.Tensor,
    input_ids: torch.Tensor,
    retrieve_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """树结构解码，生成logits和hidden state"""
    # 计算位置ID（基于输入长度偏移）
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)  # 增加batch维度

    # 树解码前向传播
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # Eagle3专属：拼接最后3层hidden states
    ea_device = model.ea_layer.lm_head.weight.device
    hidden_states = [x.to(ea_device) for x in outputs["hidden_states"]]
    hidden_state = torch.cat(hidden_states[-3:], dim=-1)

    # 按检索索引筛选logits
    logits = tree_logits[0, retrieve_indices]

    return logits, hidden_state, outputs


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList = None
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """评估候选序列的后验概率，选择最优候选"""
    # 贪心解码模式（无logits处理器）
    if logits_processor is None:
        # 生成后验掩码（候选是否匹配最大logits）
        posterior_mask = (candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)).int()
        # 计算累计接受长度
        candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = candidates_accept_length.max().item()

        # 选择最优候选
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

        return best_candidate, accept_length, logits[best_candidate, accept_length]

    # 采样模式（有logits处理器）
    accept_length = 1
    accept_cand = candidates[0][:1]
    best_candidate = 0
    adjustflag = False

    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break
        
        # 找到与当前接受候选前缀匹配的候选
        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
        
        # 处理logits并计算概率
        gt_logits = logits[fi, i-1].unsqueeze(0)
        gt_logits = logits_processor(None, gt_logits)[0]
        gtp = torch.softmax(gt_logits, dim=0)

        # 遍历候选token
        candidates_set = []
        for j in range(candidates.shape[0]):
            if not is_eq[j]:
                continue
            x = candidates[j, i]
            xi = x.item()
            if xi in candidates_set or xi == -1:
                continue
            candidates_set.append(xi)

            # 采样判断
            r = random.random()
            px = gtp[xi]
            qx = 1.0
            acp = px / qx

            if r <= acp:
                accept_cand = torch.cat([accept_cand, x.unsqueeze(0)], dim=0)
                accept_length += 1
                best_candidate = j
                break
            else:
                # 重置概率并归一化
                gtp[xi] = 0.0
                gtp = gtp / gtp.sum()
                adjustflag = True

    # 计算最终采样概率
    if adjustflag and accept_length != candidates.shape[1]:
        sample_p = gtp
    else:
        gt_logits = logits[best_candidate, accept_length - 1].unsqueeze(0)
        gt_logits = logits_processor(None, gt_logits)[0]
        sample_p = torch.softmax(gt_logits, dim=0)

    return torch.tensor(best_candidate, device=candidates.device), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
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
    """更新推理输入（input_ids、past key values等），生成新的树结构"""
    prev_input_len = input_ids.shape[1]

    # 选择最优候选的索引
    select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    select_indices = select_indices.to(input_ids.device)

    # 更新输入ID
    best_candidate_tokens = candidates[best_candidate, : accept_length + 1].unsqueeze(0).to(input_ids.device)
    input_ids = torch.cat([input_ids, best_candidate_tokens], dim=-1)

    # 更新Past Key Values
    for past_kv_data in past_key_values_data_list:
        # 源张量：候选对应的KV值
        tgt = past_kv_data[..., select_indices, :]
        # 目标张量：更新到输入长度后的位置
        dst = past_kv_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)

    # 更新current_length
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # 处理hidden state，生成新的树结构
    retrieve_hidden_state = hidden_state_new[:, retrieve_indices]
    accept_hidden_state = retrieve_hidden_state[:, best_candidate, : accept_length + 1]

    # 采样新token
    if logits_processor is not None:
        new_sample_token = torch.multinomial(sample_p, 1).unsqueeze(0)
    else:
        new_sample_token = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)

    # 生成新的树结构（Eagle3）
    new_input_ids = torch.cat([input_ids, new_sample_token.to(input_ids.device)], dim=1)
    draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = model.ea_layer.topK_genrate(
        accept_hidden_state, new_input_ids, model.base_model.lm_head, logits_processor
    )

    # 更新已生成token数
    new_token += accept_length + 1

    return (
        input_ids, draft_tokens, new_retrieve_indices, new_tree_mask,
        new_tree_position_ids, new_token, None, new_sample_token
    )


if __name__ == "__main__":
    # 测试Logits处理器
    test_logits = torch.randn(1, 5)
    test_processor = prepare_logits_processor(temperature=0.9, top_p=0.9)
    if test_processor:
        processed_logits = test_processor(None, test_logits)
        print(f"Test logits processed: {processed_logits.shape}")
    else:
        print("Test processor is None (no valid parameters)")

    # 测试路径填充
    assert pad_path([1,2,3], 5) == [1,2,3,-2,-2], "pad_path test failed"
    print("pad_path test passed")
