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
    ���ڼ��غʹ��� CIFAR-10 ���ݼ����࣬�̳��� ZeroShotDataset��
    CIFAR-10 ���ݼ����� 10 ����ͬ���� 60000 �� 32x32 ��ɫͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� CIFAR10 ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transforms)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ CIFAR-10 ���ݼ��ı�ǩ�б�

        ����:
        list[str]: CIFAR-10 ���ݼ��ı�ǩ�б�
        """
        return _labels.CIFAR10


class Caltech101(ZeroShotDataset):
    """
    ���ڼ��غʹ��� Caltech 101 ���ݼ����࣬�̳��� ZeroShotDataset��
    Caltech 101 ���ݼ����� 101 ����ͬ����ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� Caltech101 ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = caltech101.Caltech101(
            root=root, split="train" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ Caltech 101 ���ݼ��ı�ǩ�б�
        ��ǩ��ʽΪСд�ҵ���֮���ÿո�ָ���

        ����:
        list[str]: Caltech 101 ���ݼ��ı�ǩ�б�
        """
        return [" ".join(val.lower().split("_")) for val in self.dataset.categories]  # type: ignore


class OxfordFlowers(ZeroShotDataset):
    """
    ���ڼ��غʹ��� Oxford Flowers ���ݼ����࣬�̳��� ZeroShotDataset��
    Oxford Flowers ���ݼ��������ֻ�������ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� OxfordFlowers ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = oxford_flowers.OxfordFlowers(
            root=root, split="train" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ Oxford Flowers ���ݼ��ı�ǩ�б�

        ����:
        list[str]: Oxford Flowers ���ݼ��ı�ǩ�б�
        """
        return _labels.OXFORD_FLOWERS


class OxfordPets(ZeroShotDataset):
    """
    ���ڼ��غʹ��� Oxford Pets ���ݼ����࣬�̳��� ZeroShotDataset��
    Oxford Pets ���ݼ�������������ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� OxfordPets ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = datasets.OxfordIIITPet(
            root=root, split="trainval" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ Oxford Pets ���ݼ��ı�ǩ�б�

        ����:
        list[str]: Oxford Pets ���ݼ��ı�ǩ�б�
        """
        return self.dataset.classes  # type: ignore


class Food101(ZeroShotDataset):
    """
    ���ڼ��غʹ��� Food 101 ���ݼ����࣬�̳��� ZeroShotDataset��
    Food 101 ���ݼ����� 101 �ֲ�ͬʳ������ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� Food101 ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = datasets.Food101(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ Food 101 ���ݼ��ı�ǩ�б����»����滻Ϊ�ո�

        ����:
        list[str]: Food 101 ���ݼ��ı�ǩ�б�
        """
        return [val.replace("_", " ") for val in self.dataset.classes]  # type: ignore


class StanfordCars(ZeroShotDataset):
    """
    ���ڼ��غʹ��� Stanford Cars ���ݼ����࣬�̳��� ZeroShotDataset��
    Stanford Cars ���ݼ�����������������ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� StanfordCars ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = stanford_cars.StanfordCars(root_path=root, train=train, transforms=transforms)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ Stanford Cars ���ݼ��ı�ǩ�б�

        ����:
        list[str]: Stanford Cars ���ݼ��ı�ǩ�б�
        """
        return self.dataset.labels  # type: ignore


class FGVCAircraft(ZeroShotDataset):
    """
    ���ڼ��غʹ��� FGVC Aircraft ���ݼ����࣬�̳��� ZeroShotDataset��
    FGVC Aircraft ���ݼ��������ַɻ�����ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� FGVCAircraft ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = datasets.FGVCAircraft(
            root=root, download=True, split="trainval" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ FGVC Aircraft ���ݼ��ı�ǩ�б�

        ����:
        list[str]: FGVC Aircraft ���ݼ��ı�ǩ�б�
        """
        return self.dataset.classes  # type: ignore


class Imagenet(ZeroShotDataset):
    """
    ���ڼ��غʹ��� ImageNet ���ݼ����࣬�̳��� ZeroShotDataset��
    ImageNet ��һ�����ģ��ͼ�����ݼ�������������ͬ����ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� Imagenet ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ���������������֤����
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = datasets.ImageNet(
            root=root, split="train" if train else "val", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ ImageNet ���ݼ��ı�ǩ�б�

        ����:
        list[str]: ImageNet ���ݼ��ı�ǩ�б�
        """
        return self.dataset.class_to_idx.keys()  # type: ignore


class SUN397(ZeroShotDataset):
    """
    ���ڼ��غʹ��� SUN397 ���ݼ����࣬�̳��� ZeroShotDataset��
    SUN397 ���ݼ����� 397 �ֲ�ͬ��������ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� SUN397 ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = sun397.SUN397(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ SUN397 ���ݼ��ı�ǩ�б�

        ����:
        list[str]: SUN397 ���ݼ��ı�ǩ�б�
        """
        return self.dataset.categories  # type: ignore


class DTD(ZeroShotDataset):
    """
    ���ڼ��غʹ��� DTD ���ݼ����࣬�̳��� ZeroShotDataset��
    DTD ���ݼ�����������������ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� DTD ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = datasets.DTD(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ DTD ���ݼ��ı�ǩ�б�

        ����:
        list[str]: DTD ���ݼ��ı�ǩ�б�
        """
        return self.dataset.classes  # type: ignore


class EuroSAT(ZeroShotDataset):
    """
    ���ڼ��غʹ��� EuroSAT ���ݼ����࣬�̳��� ZeroShotDataset��
    EuroSAT ���ݼ���������ͼ�������������ú����ظ��Ƿ��ࡣ
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� EuroSAT ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = eurosat.EuroSAT(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ EuroSAT ���ݼ��ı�ǩ�б�

        ����:
        list[str]: EuroSAT ���ݼ��ı�ǩ�б�
        """
        return _labels.EUROSAT


class UCF101(ZeroShotDataset):
    """
    ���ڼ��غʹ��� UCF101 ���ݼ����࣬�̳��� ZeroShotDataset��
    UCF101 ���ݼ����� 101 �ֲ�ͬ���ද��������Ƶ֡ͼ��
    """
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        """
        ��ʼ�� UCF101 ���ݼ�ʵ����

        ����:
        train (bool): �Ƿ����ѵ���������Ϊ True�������ѵ������������ز��Լ���
        root (str): ���ݼ��洢�ĸ�Ŀ¼��Ĭ��Ϊ "data"��
        transforms (Callable | None): Ӧ�������ݵ�ת��������Ĭ��Ϊ None��
        """
        dataset = ucf101.UCF101(root=root, train=train, transform=transforms, download=True)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        """
        ��ȡ UCF101 ���ݼ��ı�ǩ�б����»����滻Ϊ�ո�ת��ΪСд��

        ����:
        list[str]: UCF101 ���ݼ��ı�ǩ�б�
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
    ���ݼ���ʼ������ö���࣬����ͨ�����Ƴ�ʼ����ͬ�����ݼ���
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
        �����ַ������Ʒ��ض�Ӧ�����ݼ���ʼ����ö��ֵ��

        ����:
        name (str): ���ݼ������ƣ������ִ�Сд��

        ����:
        Self: ��Ӧ�����ݼ���ʼ����ö��ֵ��
        """
        return cls[name.upper()]