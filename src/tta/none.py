import torch

from src.tta.base import BaseTTA


class NoTTA(BaseTTA):
    def before_partition(self, dataloader, move_batch_to_device, transform_batch):
        return None

    def predict_batch(self, batch):
        with torch.no_grad():
            return self.model(**batch)
