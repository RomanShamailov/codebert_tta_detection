from datasets import load_dataset
from transformers import AutoTokenizer

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CodeDataset(BaseDataset):
    """
    Code classification dataset backed by HuggingFace datasets.

    Supports local JSON/JSONL files and HuggingFace Hub datasets with `code`
    and one of `labels`, `label`, `generated`, `target`.
    """

    def __init__(
        self,
        tokenizer_name_or_path,
        name_or_path,
        source="json",
        split="test",
        subset_name=None,
        max_length=512,
        code_key="code",
        label_keys=("labels", "label", "generated", "target"),
        *args,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.code_key = code_key
        self.label_keys = tuple(label_keys)
        self.source = source
        self.split = split
        self.pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
        }

        self.dataset, dataset_id = self._load_dataset(
            source=source,
            name_or_path=name_or_path,
            split=split,
            subset_name=subset_name,
        )
        index = self._build_index(dataset_id)
        super().__init__(index, *args, **kwargs)

    def _load_dataset(self, source, name_or_path, split, subset_name):
        if source == "json":
            path = (
                ROOT_PATH / name_or_path
                if not str(name_or_path).startswith("/")
                else name_or_path
            )
            dataset = load_dataset("json", data_files={split: str(path)}, split=split)
            return dataset, str(path)

        if source == "hf":
            dataset_kwargs = {"split": split}
            if subset_name is not None:
                dataset_kwargs["name"] = subset_name
            dataset = load_dataset(name_or_path, **dataset_kwargs)
            return dataset, f"{name_or_path}:{split}"

        raise ValueError(f"Unknown code dataset source: {source}")

    def _build_index(self, dataset_id):
        index = []
        for row_idx, record in enumerate(self.dataset):
            label_key = next((key for key in self.label_keys if key in record), None)
            if label_key is None:
                raise KeyError(
                    f"No label key from {self.label_keys} in {dataset_id}:{row_idx + 1}"
                )
            index.append(
                {
                    "path": f"{dataset_id}:{row_idx}",
                    "row_idx": row_idx,
                    "label": int(record[label_key]),
                }
            )
        return index

    def __getitem__(self, ind):
        data = self._index[ind]
        record = self.dataset[data["row_idx"]]
        encoded = self.tokenizer(
            record[self.code_key],
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
        )
        instance_data = {
            "labels": data["label"],
            "index": ind,
            "pad_values": self.pad_values,
            **encoded,
        }
        return self.preprocess_data(instance_data)
