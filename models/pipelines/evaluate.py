import argparse
import logging
from typing import Any

from torch.utils.data import DataLoader

import wandb
from models import METRICS
from models._base_metric import Metric
from models import Model
from models._base_model import BaseModel
from data.datasets import DatasetInitializer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    ���������в�������������ģ����ָ�����ݼ��ϵ����ܡ�

    ����:
        argparse.Namespace: ���������������в����������ռ����
                            ���� 'model'��ģ�����ƣ��� 'datasets'�����ݼ������б���
    """
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names",
    )
    return parser.parse_args()


class EvaluatePipeline:
    def __init__(
        self, model: BaseModel, dataset_initializers: list[DatasetInitializer], metrics: list[Metric]
    ) -> None:
        """
        ��ʼ�������ܵ���

        ����:
            model (BaseModel): Ҫ������ģ�͡�
            dataset_initializers (list[DatasetInitializer]): ���ݼ���ʼ�����б����ڳ�ʼ��Ҫ���������ݼ���
            metrics (list[Metric]): ����ָ���б���������ģ�����ܡ�
        """
        self._model = model
        self._dataset_initializers = dataset_initializers
        self._metrics = metrics

    def run(self) -> dict[str, dict[str, Any]]:
        """
        ���������ܵ�����ģ����ָ�����ݼ��Ͻ���������

        ����:
            dict[str, dict[str, Any]]: ����ÿ�����ݼ�������������ֵ䣬
                                       ��Ϊ���ݼ����ƣ�ֵΪ����ÿ������ָ�꼰���Ӧֵ���ֵ䡣
        """
        results = {}

        for dataset_loader in self._dataset_initializers:
            zero_shot_dataset, dataset_name = (
                dataset_loader.value(train=False, transforms=self._model.transforms),
                dataset_loader.name,
            )
            dataset, labels = zero_shot_dataset.dataset, zero_shot_dataset.labels

            dataloader: DataLoader[Any] = DataLoader(
                dataset,
                batch_size=self._model.batch_size,
                shuffle=False,
            )
            self._model.reconfig_labels(labels)

            logger.info(f"Evaluating model on dataset {dataset.__class__.__name__}")
            y_true, y_pred = self._model.predict_for_eval(dataloader)

            for metric in self._metrics:
                metric_value = metric.calculate(y_true, y_pred)
                results[dataset_name] = {str(metric): metric_value}
                logger.info(f"Metric {metric} for dataset {dataset_name}: {metric_value}")

        return results


if __name__ == "__main__":
    args = parse_args()

    model = Model.from_str(args.model)
    dataset_initializers = [DatasetInitializer.from_str(dataset) for dataset in args.datasets]

    pipeline = EvaluatePipeline(
        model=model, dataset_initializers=dataset_initializers, metrics=METRICS  # type: ignore
    )

    wandb.init(
        project="fomo",
        config={
            "model": args.model,
            "datasets": args.datasets,
        },
    )

    results = pipeline.run()

    wandb.log(results)
    wandb.finish()