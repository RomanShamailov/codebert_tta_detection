import torch
from torch.nn.utils.rnn import pad_sequence


def _padding_value(key, dataset_items):
    pad_values = dataset_items[0].get("pad_values", {})
    return pad_values.get(key, 0)


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    keys = dataset_items[0].keys()

    for key in keys:
        if key == "pad_values":
            continue
        values = [elem[key] for elem in dataset_items]
        if isinstance(values[0], torch.Tensor):
            if values[0].ndim == 0:
                result_batch[key] = torch.stack(values)
            elif all(value.shape == values[0].shape for value in values):
                result_batch[key] = torch.stack(values)
            else:
                padding_value = _padding_value(key, dataset_items)
                result_batch[key] = pad_sequence(
                    values, batch_first=True, padding_value=padding_value
                )
        elif isinstance(values[0], int):
            result_batch[key] = torch.tensor(values)
        elif isinstance(values[0], list) and all(
            isinstance(item, int) for item in values[0]
        ):
            tensors = [torch.tensor(value, dtype=torch.long) for value in values]
            padding_value = _padding_value(key, dataset_items)
            result_batch[key] = pad_sequence(
                tensors, batch_first=True, padding_value=padding_value
            )
        else:
            result_batch[key] = values

    return result_batch
