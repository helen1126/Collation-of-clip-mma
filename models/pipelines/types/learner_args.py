import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LearnerArgs:
    """
    用于存储训练过程中各种参数配置的类。

    属性:
        use_wandb (bool): 是否使用 Weights & Biases 进行实验跟踪，默认为 True。
        epochs (int): 训练的总轮数，默认为 100。
        patience (int): 早停策略中的耐心值，默认为 5。
        model_type (str): 模型的类型，默认为 "clip_base"。
        print_freq (int): 打印训练信息的频率，默认为 200。
        save_freq (int): 保存模型的频率，默认为 200。
        output_dir (str): 输出目录，默认为 "./output/"。
        model_backbone (str): 模型的骨干网络，默认为 "ViT-B/16"。
        dataset (str): 数据集名称，默认为 "cifar10"。
        device (str): 训练设备，默认为 "cuda"。
        batch_size (int): 训练和评估时的批次大小，默认为 256。
        num_workers (int): 数据加载的工作进程数量，默认为 4。
        train_size (float | None): 训练集的大小，默认为 None。
        train_eval_size (tuple[int, int] | None): 训练评估集的大小，默认为 None。
        text_prompt_template (str): 文本提示模板，默认为 "a photo of {}."。
        learning_rate (float): 学习率，默认为 0.01。
        momentum (float): 优化器的动量，默认为 0.5。
        weight_decay (float): 权重衰减系数，默认为 1e-4。
        warmup (int): 学习率预热的步数，默认为 0。
        info (str | None): 额外信息，默认为 None。
        seed (int): 随机种子，用于保证实验的可重复性，默认为 42。
        train_subsample (str): 训练集的子采样策略，默认为 "all"。
        test_subsample (str): 测试集的子采样策略，默认为 "all"。
        gaussian_noise_std (float): 高斯噪声的标准差，默认为 0.0。
        evaluate_only (bool): 是否仅进行评估，默认为 False。
    """
    use_wandb: bool = True
    epochs: int = 100
    patience: int = 5
    model_type: str = "clip_base"
    print_freq: int = 200
    save_freq: int = 200
    output_dir: str = "./output/"
    model_backbone: str = "ViT-B/16"
    dataset: str = "cifar10"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 4
    train_size: float | None = None
    train_eval_size: tuple[int, int] | None = None
    text_prompt_template: str = "a photo of {}."
    learning_rate: float = 0.01
    momentum: float = 0.5
    weight_decay: float = 1e-4
    warmup: int = 0
    info: str | None = None
    seed: int = 42
    train_subsample: str = "all"
    test_subsample: str = "all"
    gaussian_noise_std: float = 0.0
    evaluate_only: bool = False

    def __post_init__(self) -> None:
        """
        在数据类初始化后自动调用的方法，用于生成运行 ID 并更新输出目录。

        功能:
            1. 根据模型类型、数据集名称和当前时间戳生成一个唯一的运行 ID。
            2. 将运行 ID 追加到输出目录中，更新输出目录的路径。
        """
        self.run_id = f"{self.model_type}_{self.dataset}_{str(int(time.time()))}".replace("/", "").lower()
        self.output_dir = os.path.join(self.output_dir, self.run_id)

    def save_config(self) -> None:
        """
        将当前配置保存为 JSON 文件。

        功能:
            1. 检查输出目录是否存在，如果不存在则创建该目录。
            2. 将当前配置转换为字典形式。
            3. 将字典形式的配置保存为 JSON 文件，文件名为 "config.json"，并使用 4 个空格进行缩进。
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            data = self.to_dict()
            json.dump(data, f, indent=4)

    def to_dict(self) -> dict[str, Any]:
        """
        将当前对象的属性转换为字典形式。

        返回:
            dict[str, Any]: 包含当前对象所有属性的字典。
        """
        return self.__dict__