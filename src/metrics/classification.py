import torch

from src.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        predictions = logits.argmax(dim=-1)
        return {
            "__metric_state__": "accuracy",
            "correct": (predictions == labels).float().sum(),
            "total": labels.numel(),
        }


class BinaryClassificationMetric(BaseMetric):
    def __init__(self, positive_label=1, eps=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_label = positive_label
        self.eps = eps

    def _confusion_counts(self, logits: torch.Tensor, labels: torch.Tensor):
        predictions = logits.argmax(dim=-1)
        positives = predictions == self.positive_label
        true_positives = labels == self.positive_label

        tp = (positives & true_positives).float().sum()
        fp = (positives & ~true_positives).float().sum()
        fn = (~positives & true_positives).float().sum()
        return tp, fp, fn


class Precision(BinaryClassificationMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        tp, fp, _ = self._confusion_counts(logits, labels)
        return {"__metric_state__": "precision", "tp": tp, "fp": fp}


class Recall(BinaryClassificationMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        tp, _, fn = self._confusion_counts(logits, labels)
        return {"__metric_state__": "recall", "tp": tp, "fn": fn}


class F1(BinaryClassificationMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        tp, fp, fn = self._confusion_counts(logits, labels)
        return {"__metric_state__": "f1", "tp": tp, "fp": fp, "fn": fn}
