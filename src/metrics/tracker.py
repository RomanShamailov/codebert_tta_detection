import pandas as pd
import torch


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Stored for compatibility with trainer-style code. Inference
                logging is handled by Inferencer after metrics are updated.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "value"])
        self._states = {}
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        self._states = {}

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        if isinstance(value, dict) and "__metric_state__" in value:
            self._update_state(key, value)
            return

        if isinstance(value, torch.Tensor):
            value = value.item()

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "value"] = self._data.total[key] / self._data.counts[key]

    def _update_state(self, key, value):
        state_type = value["__metric_state__"]
        state = self._states.setdefault(key, {"__metric_state__": state_type})

        for state_key, state_value in value.items():
            if state_key == "__metric_state__":
                continue
            if isinstance(state_value, torch.Tensor):
                state_value = state_value.item()
            state[state_key] = state.get(state_key, 0.0) + state_value

        self._data.loc[key, "counts"] += 1
        self._data.loc[key, "value"] = self._compute_state_metric(state)

    @staticmethod
    def _compute_state_metric(state):
        eps = 1e-8
        state_type = state["__metric_state__"]

        if state_type == "accuracy":
            return state.get("correct", 0.0) / (state.get("total", 0.0) + eps)

        tp = state.get("tp", 0.0)
        fp = state.get("fp", 0.0)
        fn = state.get("fn", 0.0)

        if state_type == "precision":
            return tp / (tp + fp + eps)
        if state_type == "recall":
            return tp / (tp + fn + eps)
        if state_type == "f1":
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            return 2 * precision * recall / (precision + recall + eps)

        raise ValueError(f"Unknown metric state type: {state_type}")

    def avg(self, key):
        """
        Return current value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            metric_value (float): current value for the metric.
        """
        return self.value(key)

    def value(self, key):
        """
        Return current value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            metric_value (float): current value for the metric.
        """
        return self._data.value[key]

    def result(self):
        """
        Return current value of each metric.

        Returns:
            metrics (dict): dict, containing current metric values
                for each metric name.
        """
        return dict(self._data.value)

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
