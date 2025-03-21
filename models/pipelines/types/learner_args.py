import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LearnerArgs:
    """
    ���ڴ洢ѵ�������и��ֲ������õ��ࡣ

    ����:
        use_wandb (bool): �Ƿ�ʹ�� Weights & Biases ����ʵ����٣�Ĭ��Ϊ True��
        epochs (int): ѵ������������Ĭ��Ϊ 100��
        patience (int): ��ͣ�����е�����ֵ��Ĭ��Ϊ 5��
        model_type (str): ģ�͵����ͣ�Ĭ��Ϊ "clip_base"��
        print_freq (int): ��ӡѵ����Ϣ��Ƶ�ʣ�Ĭ��Ϊ 200��
        save_freq (int): ����ģ�͵�Ƶ�ʣ�Ĭ��Ϊ 200��
        output_dir (str): ���Ŀ¼��Ĭ��Ϊ "./output/"��
        model_backbone (str): ģ�͵ĹǸ����磬Ĭ��Ϊ "ViT-B/16"��
        dataset (str): ���ݼ����ƣ�Ĭ��Ϊ "cifar10"��
        device (str): ѵ���豸��Ĭ��Ϊ "cuda"��
        batch_size (int): ѵ��������ʱ�����δ�С��Ĭ��Ϊ 256��
        num_workers (int): ���ݼ��صĹ�������������Ĭ��Ϊ 4��
        train_size (float | None): ѵ�����Ĵ�С��Ĭ��Ϊ None��
        train_eval_size (tuple[int, int] | None): ѵ���������Ĵ�С��Ĭ��Ϊ None��
        text_prompt_template (str): �ı���ʾģ�壬Ĭ��Ϊ "a photo of {}."��
        learning_rate (float): ѧϰ�ʣ�Ĭ��Ϊ 0.01��
        momentum (float): �Ż����Ķ�����Ĭ��Ϊ 0.5��
        weight_decay (float): Ȩ��˥��ϵ����Ĭ��Ϊ 1e-4��
        warmup (int): ѧϰ��Ԥ�ȵĲ�����Ĭ��Ϊ 0��
        info (str | None): ������Ϣ��Ĭ��Ϊ None��
        seed (int): ������ӣ����ڱ�֤ʵ��Ŀ��ظ��ԣ�Ĭ��Ϊ 42��
        train_subsample (str): ѵ�������Ӳ������ԣ�Ĭ��Ϊ "all"��
        test_subsample (str): ���Լ����Ӳ������ԣ�Ĭ��Ϊ "all"��
        gaussian_noise_std (float): ��˹�����ı�׼�Ĭ��Ϊ 0.0��
        evaluate_only (bool): �Ƿ������������Ĭ��Ϊ False��
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
        ���������ʼ�����Զ����õķ����������������� ID ���������Ŀ¼��

        ����:
            1. ����ģ�����͡����ݼ����ƺ͵�ǰʱ�������һ��Ψһ������ ID��
            2. ������ ID ׷�ӵ����Ŀ¼�У��������Ŀ¼��·����
        """
        self.run_id = f"{self.model_type}_{self.dataset}_{str(int(time.time()))}".replace("/", "").lower()
        self.output_dir = os.path.join(self.output_dir, self.run_id)

    def save_config(self) -> None:
        """
        ����ǰ���ñ���Ϊ JSON �ļ���

        ����:
            1. ������Ŀ¼�Ƿ���ڣ�����������򴴽���Ŀ¼��
            2. ����ǰ����ת��Ϊ�ֵ���ʽ��
            3. ���ֵ���ʽ�����ñ���Ϊ JSON �ļ����ļ���Ϊ "config.json"����ʹ�� 4 ���ո����������
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            data = self.to_dict()
            json.dump(data, f, indent=4)

    def to_dict(self) -> dict[str, Any]:
        """
        ����ǰ���������ת��Ϊ�ֵ���ʽ��

        ����:
            dict[str, Any]: ������ǰ�����������Ե��ֵ䡣
        """
        return self.__dict__