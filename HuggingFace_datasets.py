from pprint import pprint
from datasets import list_datasets, load_dataset
datasets_list = list_datasets()
# FutureWarning: list_datasets is deprecated and will be removed in the next major version of datasets. Use 'huggingface_hub.list_datasets' instead.
len(datasets_list)

dataset = load_dataset('sst', spilt='train')
len(datasets_list)

pprint(dataset[0])

from datasets import list_metrics, load_metric
metrics_list = list_metrics()
# FutureWarning: list_metrics is deprecated and will be removed in the next major version of datasets. Use 'evaluate.list_evaluation_modules' instead, 
# from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
len(metrics_list)

', '.join(metrics_list)

accuracy_metric = load_metric('accuracy')
results = accuracy_metric.compute(reference=[0, 1, 0], predictions=[1, 1, 0])
print(results)