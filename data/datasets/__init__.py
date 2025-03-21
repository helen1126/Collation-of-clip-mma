import enum
from typing import Callable, Self

from torchvision import datasets

from data.datasets import (
    _labels,
    caltech101,
    eurosat,
    oxford_flowers,
    stanford_cars,
    sun397,
    ucf101,
)
from data.zero_shot_dataset import ZeroShotDataset


class CIFAR10(ZeroShotDataset):
    """
    用于加载和处理 CIFAR-10 数据集的类，继承自 ZeroShotDataset。
    CIFAR-10 数据集包含 10 个不同类别的 60000 张 32x32 彩色图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 CIFAR10 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transforms)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 CIFAR-10 数据集的标签列表。

        返回:
        list[str]: CIFAR-10 数据集的标签列表。
        """
        return _labels.CIFAR10


class Caltech101(ZeroShotDataset):
    """
    用于加载和处理 Caltech 101 数据集的类，继承自 ZeroShotDataset。
    Caltech 101 数据集包含 101 个不同类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 Caltech101 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = caltech101.Caltech101(
            root=root, split="train" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 Caltech 101 数据集的标签列表。
        标签格式为小写且单词之间用空格分隔。

        返回:
        list[str]: Caltech 101 数据集的标签列表。
        """
        return [" ".join(val.lower().split("_")) for val in self.dataset.categories]  # type: ignore


class OxfordFlowers(ZeroShotDataset):
    """
    用于加载和处理 Oxford Flowers 数据集的类，继承自 ZeroShotDataset。
    Oxford Flowers 数据集包含多种花卉类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 OxfordFlowers 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = oxford_flowers.OxfordFlowers(
            root=root, split="train" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 Oxford Flowers 数据集的标签列表。

        返回:
        list[str]: Oxford Flowers 数据集的标签列表。
        """
        return _labels.OXFORD_FLOWERS


class OxfordPets(ZeroShotDataset):
    """
    用于加载和处理 Oxford Pets 数据集的类，继承自 ZeroShotDataset。
    Oxford Pets 数据集包含宠物类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 OxfordPets 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = datasets.OxfordIIITPet(
            root=root, split="trainval" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 Oxford Pets 数据集的标签列表。

        返回:
        list[str]: Oxford Pets 数据集的标签列表。
        """
        return self.dataset.classes  # type: ignore


class Food101(ZeroShotDataset):
    """
    用于加载和处理 Food 101 数据集的类，继承自 ZeroShotDataset。
    Food 101 数据集包含 101 种不同食物类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 Food101 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = datasets.Food101(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 Food 101 数据集的标签列表，将下划线替换为空格。

        返回:
        list[str]: Food 101 数据集的标签列表。
        """
        return [val.replace("_", " ") for val in self.dataset.classes]  # type: ignore


class StanfordCars(ZeroShotDataset):
    """
    用于加载和处理 Stanford Cars 数据集的类，继承自 ZeroShotDataset。
    Stanford Cars 数据集包含多种汽车类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 StanfordCars 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = stanford_cars.StanfordCars(root_path=root, train=train, transforms=transforms)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 Stanford Cars 数据集的标签列表。

        返回:
        list[str]: Stanford Cars 数据集的标签列表。
        """
        return self.dataset.labels  # type: ignore


class FGVCAircraft(ZeroShotDataset):
    """
    用于加载和处理 FGVC Aircraft 数据集的类，继承自 ZeroShotDataset。
    FGVC Aircraft 数据集包含多种飞机类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 FGVCAircraft 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = datasets.FGVCAircraft(
            root=root, download=True, split="trainval" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 FGVC Aircraft 数据集的标签列表。

        返回:
        list[str]: FGVC Aircraft 数据集的标签列表。
        """
        return self.dataset.classes  # type: ignore


class Imagenet(ZeroShotDataset):
    """
    用于加载和处理 ImageNet 数据集的类，继承自 ZeroShotDataset。
    ImageNet 是一个大规模的图像数据集，包含大量不同类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 Imagenet 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载验证集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = datasets.ImageNet(
            root=root, split="train" if train else "val", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 ImageNet 数据集的标签列表。

        返回:
        list[str]: ImageNet 数据集的标签列表。
        """
        return self.dataset.class_to_idx.keys()  # type: ignore


class SUN397(ZeroShotDataset):
    """
    用于加载和处理 SUN397 数据集的类，继承自 ZeroShotDataset。
    SUN397 数据集包含 397 种不同场景类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 SUN397 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = sun397.SUN397(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 SUN397 数据集的标签列表。

        返回:
        list[str]: SUN397 数据集的标签列表。
        """
        return self.dataset.categories  # type: ignore


class DTD(ZeroShotDataset):
    """
    用于加载和处理 DTD 数据集的类，继承自 ZeroShotDataset。
    DTD 数据集包含多种纹理类别的图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 DTD 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = datasets.DTD(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 DTD 数据集的标签列表。

        返回:
        list[str]: DTD 数据集的标签列表。
        """
        return self.dataset.classes  # type: ignore


class EuroSAT(ZeroShotDataset):
    """
    用于加载和处理 EuroSAT 数据集的类，继承自 ZeroShotDataset。
    EuroSAT 数据集包含卫星图像，用于土地利用和土地覆盖分类。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 EuroSAT 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = eurosat.EuroSAT(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 EuroSAT 数据集的标签列表。

        返回:
        list[str]: EuroSAT 数据集的标签列表。
        """
        return _labels.EUROSAT


class UCF101(ZeroShotDataset):
    """
    用于加载和处理 UCF101 数据集的类，继承自 ZeroShotDataset。
    UCF101 数据集包含 101 种不同人类动作类别的视频帧图像。
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        初始化 UCF101 数据集实例。

        参数:
        train (bool): 是否加载训练集。如果为 True，则加载训练集；否则加载测试集。
        root (str): 数据集存储的根目录，默认为 "data"。
        transforms (Callable | None): 应用于数据的转换函数，默认为 None。
        """
        dataset = ucf101.UCF101(root=root, train=train, transform=transforms, download=True)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        获取 UCF101 数据集的标签列表，将下划线替换为空格并转换为小写。

        返回:
        list[str]: UCF101 数据集的标签列表。
        """
        return [val.replace("_", " ").lower() for val in self.dataset.categories]  # type: ignore


PROMPTS = {
    "cifar10": "a photo of a {}.",
    "imagenet": "a photo of a {}.",
    "caltech101": "a photo of a {}.",
    "oxford_pets": "a photo of a {}, a type of pet.",
    "oxford_flowers": "a photo of a {}, a type of flower.",
    "food101": "a photo of {}, a type of food.",
    "stanford_cars": "a photo of a {}.",
    "fgvc_aircraft": "a photo of a {}, a type of aircraft.",
    "sun397": "a photo of a {}.",
    "dtd": "{} texture.",
    "eurosat": "a centered satellite photo of {}.",
    "ucf101": "a photo of a person doing {}.",
}


class DatasetInitializer(enum.Enum):
    """
    数据集初始化器的枚举类，用于通过名称初始化不同的数据集。
    """
    CIFAR10 = CIFAR10
    IMAGENET = Imagenet
    STANFORD_CARS = StanfordCars
    OXFORD_FLOWERS = OxfordFlowers
    OXFORD_PETS = OxfordPets
    FOOD101 = Food101
    CALTECH101 = Caltech101
    FGVC_AIRCRAFT = FGVCAircraft
    SUN397 = SUN397
    DTD = DTD
    EUROSAT = EuroSAT
    UCF101 = UCF101

    @classmethod
    def from_str(cls, name: str) -> Self:
        """
        根据字符串名称返回对应的数据集初始化器枚举值。

        参数:
        name (str): 数据集的名称，不区分大小写。

        返回:
        Self: 对应的数据集初始化器枚举值。
        """
        return cls[name.upper()]