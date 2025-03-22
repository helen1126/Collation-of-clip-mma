from models.pipelines.train import Learner
from models.pipelines.types.learner_args import LearnerArgs
from data.datasets import DatasetInitializer, PROMPTS

NUM_OF_SAMPLES = 16

def run_experiment(model_types, train_subsamples, datasets, prompts=None, num_samples=NUM_OF_SAMPLES):
    """
    运行一系列实验，针对不同的模型类型、训练子样本和数据集组合进行训练。

    此函数会遍历所有可能的模型类型、训练子样本和数据集的组合，
    对每个组合初始化数据集、设置学习参数，并使用 Learner 类运行训练过程。

    参数:
        model_types (list): 包含要使用的模型类型的列表。例如 ['model1', 'model2']。
        train_subsamples (list): 包含训练子样本类型的列表。例如 ['base', 'full']。
        datasets (list): 包含要使用的数据集名称的列表。例如 ['dataset1', 'dataset2']。
        prompts (dict, 可选): 一个字典，键为数据集名称，值为对应的提示模板。
            如果未提供，则使用默认的 PROMPTS。
        num_samples (int, 可选): 每个类别的样本数量，默认为 NUM_OF_SAMPLES。

    返回:
        无

    异常处理:
        如果在实验运行过程中出现任何异常，会捕获该异常并打印错误信息。
    """
    if prompts is None:
        prompts = PROMPTS
    
    for dataset_name in datasets:
        for model_type in model_types:
            for train_subsample in train_subsamples:
                try:
                    print(f"Training {model_type} on {dataset_name} on {train_subsample} subsample.")
                    _dataset = DatasetInitializer.from_str(dataset_name).value(train=True)
                    labels = _dataset.labels
                    prompt_template = prompts[dataset_name]
                    num_of_classes = len(labels) // 2 if train_subsample == "base" else len(labels)
                    train_eval_size = [num_samples * num_of_classes, num_samples * num_of_classes]

                    learner_args = LearnerArgs(model_type=model_type, dataset=dataset_name)
                    learner_args.model_type = model_type
                    learner_args.train_eval_size = train_eval_size
                    learner_args.text_prompt_template = prompt_template
                    learner_args.train_subsample = train_subsample
                    learner_args.dataset = dataset_name

                    learner = Learner(learner_args)
                    learner.run()
                except Exception as e:
                    print(f"Error: failed running - {str(e)}")