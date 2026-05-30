from copy import deepcopy

import torch
from torch import nn

from src.tta.base import BaseTTA


def softmax_entropy(logits):
    probabilities = logits.softmax(dim=-1)
    log_probabilities = logits.log_softmax(dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1)


class TentTTA(BaseTTA):
    """
    TENT: entropy minimization over normalization affine parameters.

    `reset_each_batch=True` adapts every batch from the original weights.
    `reset_each_batch=False` accumulates adaptation across batches.
    """

    def __init__(
        self,
        model,
        lr=1e-5,
        steps=1,
        reset_each_batch=True,
        norm_layer_names=("LayerNorm",),
    ):
        super().__init__(model)
        self.lr = lr
        self.steps = steps
        self.reset_each_batch = reset_each_batch
        self.norm_layer_names = tuple(norm_layer_names)
        self.initial_state = deepcopy(model.state_dict())
        self._configure_model()
        self.optimizer = self._make_optimizer()

    def before_partition(self, dataloader, move_batch_to_device, transform_batch):
        self._reset_model()

    def predict_batch(self, batch):
        if self.reset_each_batch:
            self._reset_model()

        losses = []
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = softmax_entropy(outputs["logits"]).mean()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())

        with torch.no_grad():
            outputs = self.model(**batch)
            final_entropy = softmax_entropy(outputs["logits"]).mean()

        outputs["tta_logs"] = {
            "tent_entropy_loss": losses[-1].item(),
            "tent_mean_entropy_loss": torch.stack(losses).mean().item(),
            "tent_final_entropy": final_entropy.item(),
            "tent_steps": self.steps,
            "tent_reset_each_batch": float(self.reset_each_batch),
        }
        return outputs

    def _configure_model(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        trainable = 0
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm) or (
                type(module).__name__ in self.norm_layer_names
            ):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad = True
                    trainable += parameter.numel()

        if trainable == 0:
            raise ValueError("TENT found no LayerNorm affine parameters to adapt.")

    def _make_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    def _reset_model(self):
        self.model.load_state_dict(self.initial_state)
        self._configure_model()
        self.optimizer = self._make_optimizer()
