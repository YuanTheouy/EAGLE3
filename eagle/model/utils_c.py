import torch

# typing
from typing import Any, List

TOPK = 10  # topk for sparse tree


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    return path + [pad_value] * (length - len(path))

class node:
    """Eagle3 树节点类（构建树形结构）"""
    def __init__(self, parent=None, value=None, dict_key=None):
        self.parent = parent
        self.value = value
        self.depth = parent.depth + 1 if parent else 0  # 节点深度（根节点深度0）
        self.children = []  # 子节点列表
        self.dict_key = dict_key  # 节点对应的路径元组
        self.index = -1  # 非叶子节点索引（后续赋值）
        if parent:
            parent.children.append(self)

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0

    def all_index(self) -> List[int]:
        """递归获取当前节点所有祖先节点的索引（含自身）"""
        if not self.parent.parent:
            return [self.index]
        return self.parent.all_index() + [self.index] # 加号代表列表拼接


class Tree:
    """Eagle3 树结构类（基于token路径列表构建树）"""
    def __init__(self, tree_list: List[List[int]]):
        # 按路径长度+路径内容排序，确保短路径优先构建
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root = node()  # 根节点
        self.node_dic = {}  # 路径元组→节点的映射

        # 遍历路径构建树节点
        for tree_node in sorted_tree_list:
            cur_value = tree_node[-1]
            if len(tree_node) == 1:
                # 根节点的子节点
                cur_node = node(parent=self.root, value=cur_value, dict_key=tuple(tree_node))
            else:
                # 非根节点，从父节点映射中找父节点
                cur_parent = self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value, dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node

        # 为非叶子节点分配索引
        self.indexnode()

    def max_depth(self) -> int:
        """获取树的最大深度"""
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self) -> int:
        """统计非叶子节点数量"""
        return sum(1 for item in self.node_dic.values() if not item.is_leaf())

    def get_node_wchild(self) -> List[node]:
        """获取所有非叶子节点列表"""
        return [item for item in self.node_dic.values() if not item.is_leaf()]

    def indexnode(self):
        """为非叶子节点分配连续索引"""
        cur_index = 0
        for key in self.node_dic:
            cur_node = self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index = cur_index
                cur_index += 1


def generate_tree_buffers(tree_choices: List[List[int]], device="cuda") -> dict:
    """
    生成Eagle3树推理所需的核心缓冲区（注意力掩码、树索引、位置ID、重复数）
    Parameters:
    - tree_choices: token路径列表（如[[1], [1,2], [1,3]]）
    - device: 张量设备（默认cuda）
    Returns:
    - 树缓冲区字典（含attn_mask/tree_indices/position_ids/repeat_nums）
    """
    # 构建树结构
    tree = Tree(tree_choices)
    nodes_wc = tree.get_node_wchild()  # 非叶子节点列表
    tree_len = tree.num_node_wchild()  # 非叶子节点数量
    max_depth = tree.max_depth()  # 树最大深度

    # 1. 统计各深度的非叶子节点数量
    depth_counts = [0] * (max_depth - 1)
    for x in nodes_wc:
        depth_counts[x.depth - 1] += 1
    depth_counts_sum = [sum(depth_counts[:i+1]) for i in range(len(depth_counts))]

    # 2. 构建树注意力掩码（自注意力可见性）
    tree_attn_mask = torch.eye(tree_len, tree_len)
    for id, x in enumerate[node](nodes_wc):
        tree_attn_mask[id, x.all_index()] = 1  # 祖先节点可见

    # 按深度切分注意力掩码
    tree_attn_mask_list0 = [tree_attn_mask[:ml, :ml] for ml in depth_counts_sum]
    tree_attn_mask_list = []
    for id, x in enumerate[Any](tree_attn_mask_list0):
        x = x[-depth_counts[id]:]
        tree_attn_mask_list.append(x)

    # 3. 构建树索引（适配TOPK）
    tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
    repeat_nums = [[] for _ in depth_counts]
    start = 0
    for i in range(len(depth_counts)):
        bias = 0
        repeat_j = 0
        parent = None
        for j in range(depth_counts[i]):
            cur_node = nodes_wc[start + j]
            cur_parent = cur_node.parent

            # 父节点变化时更新偏置
            if j != 0 and cur_parent != parent:
                bias += 1
                parent = cur_parent
                repeat_nums[i].append(j - repeat_j)
                repeat_j = j
            else:
                parent = cur_parent if j == 0 else parent

            # 计算树索引（TOPK * 偏置 + 节点值）
            tree_indices_list[i][j] = cur_node.value + TOPK * bias

        repeat_nums[i].append(j - repeat_j + 1)
        start += depth_counts[i]

    # 4. 构建位置ID（初始化为0）
    position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

    # 整理缓冲区
    tree_buffers = {
        "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
        "tree_indices": tree_indices_list,
        "position_ids": position_ids,
        "repeat_nums": repeat_nums
    }

    # 张量移到指定设备
    tree_buffers = {
        k: [i.clone().to(device) for i in v]
        if isinstance(v[0], torch.Tensor)
        else v
        for k, v in tree_buffers.items()
    }
    return tree_buffers


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    重置Past Key Values的current_length为0（Eagle3推理时重置KV缓存）
    Parameters:
    - passed_key_values: 各层的KV缓存列表
    Returns:
    - 重置后的KV缓存列表
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


if __name__=="__main__":
    from choices import mc_sim_7b_63
    a=generate_tree_buffers(mc_sim_7b_63)
    print(a)