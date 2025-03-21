from typing import Callable

from torch.utils.data import DataLoader, Dataset

from models.clip import MODELS
from models.clip.clip_base import ClipBase
from pipelines.types.learner_args import LearnerArgs
from data.datasets import DatasetInitializer
from data.utils import split_train_val, subsample_classes

def initalize_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    lr_args: LearnerArgs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    初始化训练、验证和测试数据加载器。

    参数:
        train_dataset (Dataset): 训练数据集。
        test_dataset (Dataset): 测试数据集。
        lr_args (LearnerArgs): 包含训练参数的对象，如批次大小、工作进程数等。

    返回:
        tuple[DataLoader, DataLoader, DataLoader]: 分别为训练、验证和测试数据加载器。
    """
    train_dataset, val_dataset = split_train_val(
        train_dataset, train_size=lr_args.train_size, train_eval_samples=lr_args.train_eval_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=lr_args.batch_size,
        shuffle=True,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=lr_args.batch_size,
        shuffle=False,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=lr_args.batch_size,
        shuffle=False,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def intialize_model(model_type: str, backbone: str, device: str) -> ClipBase:
    """
    初始化指定类型和骨干网络的模型，并将其移动到指定设备。

    参数:
        model_type (str): 模型的类型，用于从 MODELS 字典中选择模型。
        backbone (str): 模型的骨干网络名称。
        device (str): 设备名称，如 "cuda"、"mps" 或其他，用于指定模型运行的设备。

    返回:
        ClipBase: 初始化并移动到指定设备的模型。
    """
    model = MODELS[model_type](backbone=backbone)
    if device == "cuda":
        model.to_cuda()
    elif device == "mps":
        model.to_mps()
    else:
        model.to_cpu()

    model.eval()
    return model


def initalize_test_dataloader_subsample(
    dataset_name: str, transforms: Callable, lr_args: LearnerArgs, test_subsample: str = "all"
) -> tuple[DataLoader, list[str]]:
    """
    初始化测试数据加载器，并对测试数据集进行子采样。

    参数:
        dataset_name (str): 数据集的名称。
        transforms (Callable): 应用于测试数据的转换函数。
        lr_args (LearnerArgs): 包含训练参数的对象，如批次大小、工作进程数等。
        test_subsample (str, 可选): 测试数据的子采样策略，默认为 "all"。

    返回:
        tuple[DataLoader, list[str]]: 子采样后的测试数据加载器和对应的标签列表。
    """
    test_zero_shot_dataset = DatasetInitializer.from_str(dataset_name).value(
        train=False, transforms=transforms
    )

    test_dataset = test_zero_shot_dataset.dataset
    test_labels = test_zero_shot_dataset.labels

    subsampled_test_dataset, test_label_idx = subsample_classes(test_dataset, subsample=test_subsample)
    test_labels = [test_labels[i] for i in test_label_idx]

    test_loader = DataLoader(
        subsampled_test_dataset,
        batch_size=lr_args.batch_size,
        shuffle=False,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )

    return test_loader, test_labels


def initalize_datasets(
    dataset_name: str,
    train_transforms: Callable,
    eval_transforms: Callable,
    train_subsample: str = "all",
    test_subsample: str = "all",
) -> tuple[tuple[Dataset, Dataset], tuple[list[str], list[str]]]:
    """
    初始化训练和测试数据集，并对它们进行子采样。

    参数:
        dataset_name (str): 数据集的名称。
        train_transforms (Callable): 应用于训练数据的转换函数。
        eval_transforms (Callable): 应用于评估数据的转换函数。
        train_subsample (str, 可选): 训练数据的子采样策略，默认为 "all"。
        test_subsample (str, 可选): 测试数据的子采样策略，默认为 "all"。

    返回:
        tuple[tuple[Dataset, Dataset], tuple[list[str], list[str]]]: 
            包含子采样后的训练和测试数据集的元组，以及对应的训练和测试标签列表的元组。
    """
    train_zero_shot_dataset = DatasetInitializer.from_str(dataset_name).value(
        train=True, transforms=train_transforms
    )
    test_zero_shot_dataset = DatasetInitializer.from_str(dataset_name).value(
        train=False, transforms=eval_transforms
    )

    train_dataset = train_zero_shot_dataset.dataset
    test_dataset = test_zero_shot_dataset.dataset
    train_labels = train_zero_shot_dataset.labels
    test_labels = test_zero_shot_dataset.labels

    subsampled_train_dataset, train_label_idxs = subsample_classes(train_dataset, subsample=train_subsample)
    subsampled_test_dataset, test_label_idx = subsample_classes(test_dataset, subsample=test_subsample)

    train_labels = [train_labels[i] for i in train_label_idxs]
    test_labels = [test_labels[i] for i in test_label_idx]

    return ((subsampled_train_dataset, subsampled_test_dataset), (train_labels, test_labels))