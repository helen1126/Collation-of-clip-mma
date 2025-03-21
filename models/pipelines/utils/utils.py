import os
import shutil
from typing import Any, Callable, List, Tuple

import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    设置随机种子以确保实验的可重复性。

    参数:
        seed (int): 要设置的随机种子值。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    计算预测标签和真实标签之间的准确率。

    参数:
        y_true (List[int]): 真实标签的列表。
        y_pred (List[int]): 预测标签的列表。

    返回:
        float: 准确率得分。
    """
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def accuracy_score_from_logits(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    从对数概率（logits）中计算准确率得分。

    参数:
        y_true (torch.Tensor): 形状为 (N, num_classes) 的张量，包含真实标签。
        y_pred (torch.Tensor): 形状为 (N, num_classes) 的张量，包含预测的对数概率。

    返回:
        float: 准确率得分的标量值。
    """
    _, y_pred = torch.max(y_pred, dim=1)
    return accuracy_score(y_true.detach().cpu().tolist(), y_pred.detach().cpu().tolist())


class AverageMeter(object):
    """
    计算并存储当前值和平均值。

    属性:
        name (str): 指标的名称。
        fmt (str): 格式化字符串，用于显示值。
        val (float): 当前值。
        avg (float): 平均值。
        sum (float): 总和。
        count (int): 样本数量。
    """

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """
        初始化 AverageMeter 类。

        参数:
            name (str): 指标的名称。
            fmt (str, 可选): 格式化字符串，用于显示值，默认为 ":f"。
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """
        重置所有存储的值。
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        更新当前值和平均值。

        参数:
            val (float): 当前值。
            n (int, 可选): 样本数量，默认为 1。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """
        返回格式化后的字符串，包含当前值和平均值。

        返回:
            str: 格式化后的字符串。
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def assign_learning_rate(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """
    为优化器的所有参数组分配新的学习率。

    参数:
        optimizer (torch.optim.Optimizer): 要更新学习率的优化器。
        new_lr (float): 新的学习率。
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    """
    计算热身阶段的学习率。

    参数:
        base_lr (float): 基础学习率。
        warmup_length (int): 热身阶段的步数。
        step (int): 当前步数。

    返回:
        float: 热身阶段的学习率。
    """
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer: torch.optim.Optimizer, base_lr: float, warmup_length: int, steps: int) -> Callable:
    """
    创建一个学习率调整函数，使用余弦退火策略。

    参数:
        optimizer (torch.optim.Optimizer): 要调整学习率的优化器。
        base_lr (float): 基础学习率。
        warmup_length (int): 热身阶段的步数。
        steps (int): 总步数。

    返回:
        Callable: 学习率调整函数，接受当前步数作为输入，返回调整后的学习率。
    """
    def _lr_adjuster(step: int) -> float:
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[torch.Tensor]:
    """
    计算指定 k 值下的前 k 准确率。

    参数:
        output (torch.Tensor): 模型的输出张量。
        target (torch.Tensor): 真实标签的张量。
        topk (Tuple[int], 可选): 要计算准确率的 k 值元组，默认为 (1,)。

    返回:
        List[torch.Tensor]: 包含每个 k 值对应的准确率的张量列表。
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    """
    用于显示训练进度和指标的类。

    属性:
        batch_fmtstr (str): 批次格式化字符串。
        meters (List[object]): 存储指标的对象列表。
        prefix (str): 显示信息的前缀。
    """

    def __init__(self, num_batches: int, meters: List[object], prefix: str = "") -> None:
        """
        初始化 ProgressMeter 类。

        参数:
            num_batches (int): 总批次数。
            meters (List[object]): 存储指标的对象列表。
            prefix (str, 可选): 显示信息的前缀，默认为 ""。
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """
        显示当前批次的进度和指标信息。

        参数:
            batch (int): 当前批次号。
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """
        生成批次格式化字符串。

        参数:
            num_batches (int): 总批次数。

        返回:
            str: 批次格式化字符串。
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_checkpoint(
    state: Any,
    model_folder: str,
    is_best: bool = False,
    filename: str = "checkpoint.pth.tar",
) -> None:
    """
    保存模型的检查点文件。

    参数:
        state (Any): 要保存的模型状态。
        model_folder (str): 保存模型的文件夹路径。
        is_best (bool, 可选): 是否为最佳模型，默认为 False。
        filename (str, 可选): 检查点文件的名称，默认为 "checkpoint.pth.tar"。
    """
    savefile = os.path.join(model_folder, filename)
    bestfile = os.path.join(model_folder, "model_best.pth.tar")
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print("saved best file")


def build_label_prompts(labels: list[str], prompt_template: str) -> list[str]:
    """
    根据标签列表和提示模板构建提示列表。

    参数:
        labels (list[str]): 标签列表。
        prompt_template (str): 提示模板字符串，包含占位符 {}。

    返回:
        list[str]: 构建好的提示列表。
    """
    return [prompt_template.format(label) for label in labels]