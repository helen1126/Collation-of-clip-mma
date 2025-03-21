from models.pipelines.train import Learner
from models.pipelines.types.learner_args import LearnerArgs
from data.datasets import DatasetInitializer, PROMPTS

NUM_OF_SAMPLES = 16

def run_experiment(model_types, train_subsamples, datasets, prompts=None, num_samples=NUM_OF_SAMPLES):
    """
    ����һϵ��ʵ�飬��Բ�ͬ��ģ�����͡�ѵ�������������ݼ���Ͻ���ѵ����

    �˺�����������п��ܵ�ģ�����͡�ѵ�������������ݼ�����ϣ�
    ��ÿ����ϳ�ʼ�����ݼ�������ѧϰ��������ʹ�� Learner ������ѵ�����̡�

    ����:
        model_types (list): ����Ҫʹ�õ�ģ�����͵��б����� ['model1', 'model2']��
        train_subsamples (list): ����ѵ�����������͵��б����� ['base', 'full']��
        datasets (list): ����Ҫʹ�õ����ݼ����Ƶ��б����� ['dataset1', 'dataset2']��
        prompts (dict, ��ѡ): һ���ֵ䣬��Ϊ���ݼ����ƣ�ֵΪ��Ӧ����ʾģ�塣
            ���δ�ṩ����ʹ��Ĭ�ϵ� PROMPTS��
        num_samples (int, ��ѡ): ÿ����������������Ĭ��Ϊ NUM_OF_SAMPLES��

    ����:
        ��

    �쳣����:
        �����ʵ�����й����г����κ��쳣���Ჶ����쳣����ӡ������Ϣ��
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