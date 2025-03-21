import os
import shutil
from typing import Any, Callable, List, Tuple

import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    �������������ȷ��ʵ��Ŀ��ظ��ԡ�

    ����:
        seed (int): Ҫ���õ��������ֵ��
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
    ����Ԥ���ǩ����ʵ��ǩ֮���׼ȷ�ʡ�

    ����:
        y_true (List[int]): ��ʵ��ǩ���б�
        y_pred (List[int]): Ԥ���ǩ���б�

    ����:
        float: ׼ȷ�ʵ÷֡�
    """
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def accuracy_score_from_logits(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    �Ӷ������ʣ�logits���м���׼ȷ�ʵ÷֡�

    ����:
        y_true (torch.Tensor): ��״Ϊ (N, num_classes) ��������������ʵ��ǩ��
        y_pred (torch.Tensor): ��״Ϊ (N, num_classes) ������������Ԥ��Ķ������ʡ�

    ����:
        float: ׼ȷ�ʵ÷ֵı���ֵ��
    """
    _, y_pred = torch.max(y_pred, dim=1)
    return accuracy_score(y_true.detach().cpu().tolist(), y_pred.detach().cpu().tolist())


class AverageMeter(object):
    """
    ���㲢�洢��ǰֵ��ƽ��ֵ��

    ����:
        name (str): ָ������ơ�
        fmt (str): ��ʽ���ַ�����������ʾֵ��
        val (float): ��ǰֵ��
        avg (float): ƽ��ֵ��
        sum (float): �ܺ͡�
        count (int): ����������
    """

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """
        ��ʼ�� AverageMeter �ࡣ

        ����:
            name (str): ָ������ơ�
            fmt (str, ��ѡ): ��ʽ���ַ�����������ʾֵ��Ĭ��Ϊ ":f"��
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """
        �������д洢��ֵ��
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        ���µ�ǰֵ��ƽ��ֵ��

        ����:
            val (float): ��ǰֵ��
            n (int, ��ѡ): ����������Ĭ��Ϊ 1��
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """
        ���ظ�ʽ������ַ�����������ǰֵ��ƽ��ֵ��

        ����:
            str: ��ʽ������ַ�����
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def assign_learning_rate(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """
    Ϊ�Ż��������в���������µ�ѧϰ�ʡ�

    ����:
        optimizer (torch.optim.Optimizer): Ҫ����ѧϰ�ʵ��Ż�����
        new_lr (float): �µ�ѧϰ�ʡ�
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    """
    ��������׶ε�ѧϰ�ʡ�

    ����:
        base_lr (float): ����ѧϰ�ʡ�
        warmup_length (int): ����׶εĲ�����
        step (int): ��ǰ������

    ����:
        float: ����׶ε�ѧϰ�ʡ�
    """
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer: torch.optim.Optimizer, base_lr: float, warmup_length: int, steps: int) -> Callable:
    """
    ����һ��ѧϰ�ʵ���������ʹ�������˻���ԡ�

    ����:
        optimizer (torch.optim.Optimizer): Ҫ����ѧϰ�ʵ��Ż�����
        base_lr (float): ����ѧϰ�ʡ�
        warmup_length (int): ����׶εĲ�����
        steps (int): �ܲ�����

    ����:
        Callable: ѧϰ�ʵ������������ܵ�ǰ������Ϊ���룬���ص������ѧϰ�ʡ�
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
    ����ָ�� k ֵ�µ�ǰ k ׼ȷ�ʡ�

    ����:
        output (torch.Tensor): ģ�͵����������
        target (torch.Tensor): ��ʵ��ǩ��������
        topk (Tuple[int], ��ѡ): Ҫ����׼ȷ�ʵ� k ֵԪ�飬Ĭ��Ϊ (1,)��

    ����:
        List[torch.Tensor]: ����ÿ�� k ֵ��Ӧ��׼ȷ�ʵ������б�
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
    ������ʾѵ�����Ⱥ�ָ����ࡣ

    ����:
        batch_fmtstr (str): ���θ�ʽ���ַ�����
        meters (List[object]): �洢ָ��Ķ����б�
        prefix (str): ��ʾ��Ϣ��ǰ׺��
    """

    def __init__(self, num_batches: int, meters: List[object], prefix: str = "") -> None:
        """
        ��ʼ�� ProgressMeter �ࡣ

        ����:
            num_batches (int): ����������
            meters (List[object]): �洢ָ��Ķ����б�
            prefix (str, ��ѡ): ��ʾ��Ϣ��ǰ׺��Ĭ��Ϊ ""��
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """
        ��ʾ��ǰ���εĽ��Ⱥ�ָ����Ϣ��

        ����:
            batch (int): ��ǰ���κš�
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """
        �������θ�ʽ���ַ�����

        ����:
            num_batches (int): ����������

        ����:
            str: ���θ�ʽ���ַ�����
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
    ����ģ�͵ļ����ļ���

    ����:
        state (Any): Ҫ�����ģ��״̬��
        model_folder (str): ����ģ�͵��ļ���·����
        is_best (bool, ��ѡ): �Ƿ�Ϊ���ģ�ͣ�Ĭ��Ϊ False��
        filename (str, ��ѡ): �����ļ������ƣ�Ĭ��Ϊ "checkpoint.pth.tar"��
    """
    savefile = os.path.join(model_folder, filename)
    bestfile = os.path.join(model_folder, "model_best.pth.tar")
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print("saved best file")


def build_label_prompts(labels: list[str], prompt_template: str) -> list[str]:
    """
    ���ݱ�ǩ�б����ʾģ�幹����ʾ�б�

    ����:
        labels (list[str]): ��ǩ�б�
        prompt_template (str): ��ʾģ���ַ���������ռλ�� {}��

    ����:
        list[str]: �����õ���ʾ�б�
    """
    return [prompt_template.format(label) for label in labels]