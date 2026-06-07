from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from src.tta.base import BaseTTA
from src.tta.tent import softmax_entropy


class T2ATTA(BaseTTA):
    """
    Think Twice before Adaptation (T2A-lite) for binary CodeBERT detection.

    This adapts the T2A idea to transformer text/code classification:
    entropy minimization is combined with uncertainty-aware negative learning,
    while trainable parameters are restricted to normalization affine weights
    (LayerNorm by default), matching the existing TENT setup.

    The BatchNorm-gradient masking component from the original image/deepfake
    paper is intentionally not implemented here because CodeBERT uses LayerNorm.
    """

    def __init__(
        self,
        model,
        lr=1e-6,
        steps=1,
        reset_each_batch=True,
        entropy_weight=1.0,
        negative_weight=1.0,
        passive_weight=1.0,
        gamma=2.0,
        num_classes=2,
        threshold=0.5,
        eps=1e-8,
        norm_layer_names=("LayerNorm",),
        generator_seed=42,
    ):
        if num_classes != 2:
            raise ValueError("T2ATTA currently supports binary classification only.")

        super().__init__(model)
        self.lr = lr
        self.steps = steps
        self.reset_each_batch = reset_each_batch
        self.entropy_weight = entropy_weight
        self.negative_weight = negative_weight
        self.passive_weight = passive_weight
        self.gamma = gamma
        self.num_classes = num_classes
        self.threshold = threshold
        self.eps = eps
        self.norm_layer_names = tuple(norm_layer_names)
        self.generator_seed = generator_seed

        self.initial_state = deepcopy(model.state_dict())
        self.generator = None
        self._configure_model()
        self.optimizer = self._make_optimizer()

    def before_partition(self, dataloader, move_batch_to_device, transform_batch):
        self._reset_model()
        device = next(self.model.parameters()).device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.generator_seed)
        return {
            "t2a_lr": self.lr,
            "t2a_steps": self.steps,
            "t2a_reset_each_batch": float(self.reset_each_batch),
            "t2a_entropy_weight": self.entropy_weight,
            "t2a_negative_weight": self.negative_weight,
            "t2a_passive_weight": self.passive_weight,
            "t2a_gamma": self.gamma,
            "t2a_threshold": self.threshold,
        }

    def predict_batch(self, batch):
        if self.reset_each_batch:
            self._reset_model()

        loss_logs = []
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            logits = outputs["logits"]

            entropy_loss = softmax_entropy(logits).mean()
            t2a_loss, logs = self._t2a_negative_loss(logits)
            loss = self.entropy_weight * entropy_loss + t2a_loss

            loss.backward()
            self.optimizer.step()
            loss_logs.append(
                {
                    "loss": loss.detach(),
                    "entropy": entropy_loss.detach(),
                    "negative": logs["negative_loss"].detach(),
                    "normalized_negative": logs["normalized_negative_loss"].detach(),
                    "passive": logs["passive_loss"].detach(),
                    "flip_rate": logs["flip_rate"].detach(),
                    "confidence": logs["confidence"].detach(),
                }
            )

        with torch.no_grad():
            outputs = self.model(**batch)
            final_entropy = softmax_entropy(outputs["logits"]).mean()
            final_confidence = outputs["logits"].softmax(dim=-1).max(dim=-1).values.mean()

        outputs["tta_logs"] = self._aggregate_logs(loss_logs)
        outputs["tta_logs"].update(
            {
                "t2a_final_entropy": final_entropy.item(),
                "t2a_final_confidence": final_confidence.item(),
                "t2a_steps": self.steps,
                "t2a_reset_each_batch": float(self.reset_each_batch),
            }
        )
        return outputs

    def _t2a_negative_loss(self, logits):
        probabilities = logits.softmax(dim=-1)
        class_one_probability = probabilities[:, 1]
        pseudo_labels = (class_one_probability >= self.threshold).long()
        confidence = torch.where(
            pseudo_labels == 1,
            class_one_probability,
            1.0 - class_one_probability,
        )

        random_values = torch.rand(
            confidence.shape,
            device=confidence.device,
            generator=self.generator,
        )
        should_flip = random_values < (1.0 - confidence)
        noisy_labels = torch.where(should_flip, 1 - pseudo_labels, pseudo_labels)

        per_class_focal_ce = self._per_class_focal_ce(logits)
        selected_loss = per_class_focal_ce.gather(
            dim=1,
            index=noisy_labels.unsqueeze(1),
        ).squeeze(1)
        normalized_negative_loss = (
            selected_loss / per_class_focal_ce.sum(dim=1).clamp_min(self.eps)
        ).mean()

        # Passive term from the T2A paper adapted to focal CE values. This term
        # counteracts underfitting of the normalized loss under noisy labels.
        p0 = probabilities.min(dim=-1, keepdim=True).values.detach()
        passive_denominator = (p0 - per_class_focal_ce).sum(dim=1).clamp_max(-self.eps)
        passive_loss = (1.0 - (p0.squeeze(1) - selected_loss) / passive_denominator).mean()

        negative_loss = (
            self.negative_weight * normalized_negative_loss
            + self.passive_weight * passive_loss
        )

        return negative_loss, {
            "negative_loss": negative_loss,
            "normalized_negative_loss": normalized_negative_loss,
            "passive_loss": passive_loss,
            "flip_rate": should_flip.float().mean(),
            "confidence": confidence.mean(),
        }

    def _per_class_focal_ce(self, logits):
        probabilities = logits.softmax(dim=-1).clamp_min(self.eps)
        log_probabilities = probabilities.log()
        focal_weight = (1.0 - probabilities).pow(self.gamma)
        return -focal_weight * log_probabilities

    @staticmethod
    def _aggregate_logs(logs):
        result = {}
        for key in logs[0]:
            values = torch.stack([step_logs[key] for step_logs in logs])
            result[f"t2a_{key}"] = values[-1].item()
            result[f"t2a_mean_{key}"] = values.mean().item()
        return result

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
            raise ValueError("T2A found no normalization affine parameters to adapt.")

    def _make_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    def _reset_model(self):
        self.model.load_state_dict(self.initial_state)
        self._configure_model()
        self.optimizer = self._make_optimizer()
