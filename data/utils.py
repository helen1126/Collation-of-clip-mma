from typing import Any

import torch
import torch.utils
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader


class RemappedDataset(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int], label_mapping: dict):
        """
        初始化RemappedDataset类的实例。

        参数:
            dataset (torch.utils.data.Dataset): 原始数据集。
            indices (list[int]): 用于从原始数据集中选择样本的索引列表。
            label_mapping (dict): 原始标签到新标签的映射字典。
        """
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        """
        获取RemappedDataset的长度，即样本数量。

        返回:
            int: 数据集中样本的数量，由索引列表的长度决定。
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """
        根据给定的索引从RemappedDataset中获取一个样本。

        参数:
            idx (int): 样本的索引。

        返回:
            tuple[Any, int]: 包含图像和重映射后标签的元组。
        """
        original_idx = self.indices[idx]
        img, original_label = self.dataset[original_idx]
        new_label = self.label_mapping[original_label]
        return img, new_label


def get_all_labels(dataset: Dataset, batch_size: int = 2048) -> list:
    """
    从给定的数据集中获取所有标签。

    参数:
        dataset (Dataset): 要提取标签的数据集。
        batch_size (int): 数据加载器的批量大小，默认为2048。

    返回:
        list: 包含数据集中所有标签的列表。
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels)
    return all_labels


def subsample_classes(dataset: Dataset, subsample: str = "all") -> tuple[Dataset, list[str]]:
    """
    根据提供的子采样参数对数据集进行子采样。

    参数:
        dataset (Dataset): 要进行子采样的原始数据集。
        subsample (str): 子采样的类型，可选值为 'all', 'base', 'new'，默认为 'all'。
            - 'all': 返回整个数据集。
            - 'base': 返回基础类（类别的前半部分）。
            - 'new': 返回新类（类别的后半部分）。

    返回:
        tuple[Dataset, list[str]]: 包含子采样后的数据集和所选类别的元组。

    异常:
        ValueError: 如果子采样参数不是 'all', 'base', 或 'new'，则抛出此异常。
    """
    try:
        y_labels = dataset.targets
    except Exception as e:
        y_labels = [dataset[i][1] for i in range(len(dataset))]  # type: ignore
    samples_idxs = range(len(dataset))  # type: ignore
    unique_labels = sorted(set(y_labels))

    base_classes = unique_labels[: len(unique_labels) // 2]
    new_classes = unique_labels[len(unique_labels) // 2 :]
    #base_classes = unique_labels[: (len(unique_labels) - len(unique_labels) // 5)]
    #new_classes = unique_labels[(len(unique_labels) - len(unique_labels) // 5) :]

    if subsample == "all":
        selected_classes = unique_labels
    elif subsample == "base":
        selected_classes = base_classes
    elif subsample == "new":
        selected_classes = new_classes
    else:
        raise ValueError("Subsample must be one of 'all', 'base', or 'new'.")

    # Create a mapping from original labels to new contiguous labels
    label_mapping = {original: new for new, original in enumerate(selected_classes)}

    # Filter the indices and remap the labels
    subsample_idxs = [i for i in samples_idxs if y_labels[i] in selected_classes]

    # Return the subset of the dataset with remapped labels
    remapped_dataset = RemappedDataset(dataset, subsample_idxs, label_mapping)
    return remapped_dataset, selected_classes


def split_train_val(
    dataset: Dataset,
    train_size: float | None = None,
    train_eval_samples: tuple[int, int] | None = None,
) -> list[Subset[Any]]:
    """
    根据提供的参数将数据集划分为训练集和验证集。

    如果提供了 train_size，则根据 train_size 将数据集划分为训练集和验证集。
    如果提供了 train_eval_samples，则使用分层采样根据每个集合的样本数量将数据集划分为训练集和验证集。

    参数:
        dataset (Dataset): 要划分的原始数据集。
        train_size (float | None): 训练集的比例，默认为 None。
        train_eval_samples (tuple[int, int] | None): 训练集和验证集的样本数量元组，默认为 None。

    返回:
        list[Subset[Any]]: 包含训练集和验证集子集的列表。

    异常:
        ValueError: 如果 train_size 和 train_eval_samples 都未提供，则抛出此异常。
    """
    if train_size is not None:
        train_samples = int(train_size * float(len(dataset)))  # type: ignore
        val_samples = len(dataset) - train_samples  # type: ignore
        return torch.utils.data.random_split(dataset, [train_samples, val_samples])
    elif train_eval_samples is not None:
        train_idx, val_idx = _get_train_val_idx(dataset, train_eval_samples)
        return [Subset(dataset, train_idx), Subset(dataset, val_idx)]
    else:
        raise ValueError("Either train_size or train_eval_samples must be provided.")


def _get_train_val_idx(dataset: Dataset, train_eval_samples: tuple[int, int]) -> tuple[list[int], list[int]]:
    """
    获取用于划分训练集和验证集的索引。

    参数:
        dataset (Dataset): 要划分的原始数据集。
        train_eval_samples (tuple[int, int]): 训练集和验证集的样本数量元组。

    返回:
        tuple[list[int], list[int]]: 包含训练集和验证集索引的元组。

    异常:
        ValueError: 如果训练集和验证集的样本数量不能被类别数量整除，则抛出此异常。
    """
    y_labels = [dataset[i][1] for i in range(len(dataset))]  # type: ignore
    class_count = len(set(y_labels))
    train_samples, val_samples = train_eval_samples

    if train_samples % class_count != 0 or val_samples % class_count != 0:
        raise ValueError("train_samples and val_samples must be divisible by the number of classes.")

    train_idx, val_idx = train_test_split(
        range(len(dataset)),  # type: ignore
        stratify=y_labels,
        train_size=train_samples,
        test_size=val_samples,
    )

    return train_idx, val_idx