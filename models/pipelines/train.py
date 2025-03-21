import argparse
import time

import torch
import torch.nn as nn
import torch.utils
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from utils.add_gaussian_noise import AddGaussianNoise

try:
    from wandb import wandb
except ImportError:
    print("Wandb is not installed. Please install it to log the results.")

import logging

from models.pipelines.types.learner_args import LearnerArgs
from models.pipelines.utils.initializers import (
    initalize_dataloaders,
    initalize_datasets,
    initalize_test_dataloader_subsample,
    intialize_model,
)
from models.pipelines.utils.utils import (
    AverageMeter,
    ProgressMeter,
    accuracy,
    build_label_prompts,
    cosine_lr,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    此函数创建一个命令行参数解析器，允许用户指定是否使用 wandb 进行日志记录，以及要使用的数据集名称。

    返回:
        argparse.Namespace: 包含解析后的命令行参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use wandb for logging",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names",
    )
    return parser.parse_args()

class Learner:
    """
    使用 CLIP 进行提示学习的训练器。

    该类封装了训练和评估模型的整个流程，包括数据加载、模型初始化、训练循环、评估和保存检查点等操作。
    """

    def __init__(self, learner_args: LearnerArgs) -> None:
        """
        初始化 Learner 类的实例。

        参数:
            learner_args (LearnerArgs): 包含训练所需参数的对象。
        """
        self._lr_args = learner_args

        set_seed(self._lr_args.seed)
        self.best_acc1 = 0.0

        self.model = intialize_model(
            self._lr_args.model_type, self._lr_args.model_backbone, self._lr_args.device
        )

        self.train_transforms = self.model.transforms
        self.eval_transforms = self.model.transforms

        if self._lr_args.gaussian_noise_std > 0:
            noise_transform = AddGaussianNoise(0.0, self._lr_args.gaussian_noise_std)
            self.train_transforms.transforms.append(noise_transform)

        (train_dataset, test_dataset), (train_labels, test_labels) = initalize_datasets(
            dataset_name=self._lr_args.dataset,
            train_transforms=self.train_transforms,
            eval_transforms=self.eval_transforms,
            train_subsample=self._lr_args.train_subsample,
            test_subsample=self._lr_args.test_subsample,
        )

        self.train_labels = train_labels
        self.test_labels = test_labels

        self.train_loader, self.val_loader, self.test_loader = initalize_dataloaders(
            train_dataset, test_dataset, self._lr_args
        )

        self.criterion = nn.CrossEntropyLoss()

        if self._lr_args.evaluate_only:
            return

        self._configure_trainable_params()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self._lr_args.learning_rate,
            weight_decay=self._lr_args.weight_decay,
        )

        self.scaler = GradScaler()

        self.scheduler = cosine_lr(
            self.optimizer,
            self._lr_args.learning_rate,
            self._lr_args.warmup,
            len(self.train_loader) * self._lr_args.epochs,
        )

    def _configure_trainable_params(self) -> None:
        """
        配置模型的可训练参数。

        此方法将模型中除了指定的可学习参数外的所有参数的梯度设置为不可训练，
        并打印出可训练的参数名称和可训练参数的数量。
        """
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for learnable_param_name in self.model.learnable_param_names:
                if learnable_param_name in name:
                    param.requires_grad = True

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of learnable paramms: {num_params}")

    def run(self) -> None:
        """
        运行训练过程。

        该方法会执行完整的训练流程，包括保存配置、初始化 wandb（如果需要）、
        训练多个 epoch、评估模型、保存检查点以及最终在测试集上进行评估。
        """
        self._lr_args.save_config()

        if self._lr_args.use_wandb:
            wandb.init(
                project="fomo",
                name=self._lr_args.run_id,
                config=self._lr_args.to_dict(),
                reinit=True,
            )

        if self._lr_args.evaluate_only:
            self.evaluate("test_all")
            self.evaluate("test_base")
            self.evaluate("test_new")
            return

        epochs_since_improvement = 0

        for epoch in range(self._lr_args.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            acc1 = self.evaluate()

            if self._lr_args.use_wandb:
                wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": acc1})

            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_acc1": self.best_acc1,
                    "optimizer": self.optimizer.state_dict(),
                },
                self._lr_args.output_dir,
                is_best=is_best,
            )

            if is_best:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                print(f"There's no improvement for {epochs_since_improvement} epochs.")

                if epochs_since_improvement >= self._lr_args.patience:
                    print("The training halted by early stopping criterion.")
                    break

        self.model.load_state_dict(torch.load(f"{self._lr_args.output_dir}/model_best.pth.tar")["state_dict"])

        self.evaluate("test_all")
        self.evaluate("test_base")
        self.evaluate("test_new")

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """
        训练一个 epoch。

        该方法用于在一个 epoch 内对模型进行训练，包括前向传播、反向传播、更新参数等操作，
        并记录训练过程中的损失和准确率。

        参数:
            epoch (int): 当前的 epoch 编号。

        返回:
            tuple[float, float]: 一个元组，包含该 epoch 的平均损失和平均准确率。
        """
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch),
        )

        self.model.train()

        self.model.precompute_prompt_features(
            build_label_prompts(self.train_labels, self._lr_args.text_prompt_template)
        )

        num_batches_per_epoch = len(self.train_loader)

        end = time.time()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            data_time.update(time.time() - end)

            # step = num_batches_per_epoch * epoch + i
            # self.scheduler(step)

            self.optimizer.zero_grad()
            images = images.to(self.model.device)
            targets = targets.to(self.model.device)
            outputs = self.model(images).squeeze(-1)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)

            acc1 = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self._lr_args.print_freq == 0:
                progress.display(i)

            if i % self._lr_args.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_acc1": self.best_acc1,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self._lr_args.output_dir,
                )

        return losses.avg, top1.avg

    def evaluate(self, split: str = "valid") -> float:
        """
        在指定的数据集划分上评估模型。

        该方法用于在验证集或测试集上评估模型的性能，并返回平均准确率。

        参数:
            split (str, 可选): 要评估的数据集划分，默认为 "valid"。

        返回:
            float: 评估的平均准确率。
        """
        if split == "valid":
            loader = self.val_loader
        else:
            test_subsample = split.split("_")[-1]
            loader, test_labels = initalize_test_dataloader_subsample(
                self._lr_args.dataset, self.eval_transforms, self._lr_args, test_subsample
            )

            self.model.precompute_prompt_features(
                build_label_prompts(test_labels, self._lr_args.text_prompt_template)
            )

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1_prompt = AverageMeter("Prompt Acc@1", ":6.2f")

        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1_prompt],
            prefix="Validate: ",
        )

        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.to(self.model.device)
                target = target.to(self.model.device)

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1 = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_prompt.update(acc1[0].item(), images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self._lr_args.print_freq == 0:
                    progress.display(i)

            print(" * Prompt Acc@1 {top1_prompt.avg:.3f}".format(top1_prompt=top1_prompt))

        if self._lr_args.use_wandb:
            wandb.log({f"{split}_acc": top1_prompt.avg})

        return top1_prompt.avg