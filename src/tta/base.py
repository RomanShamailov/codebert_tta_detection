from abc import ABC, abstractmethod


class BaseTTA(ABC):
    """
    Common interface for test-time adaptation methods.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def before_partition(self, dataloader, move_batch_to_device, transform_batch):
        """
        Prepare adapter state before processing a dataset partition.

        Returns:
            dict | None: scalar logs describing the preparation stage.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, batch):
        """
        Return model outputs for one already prepared batch.
        Implementations may include a `tta_logs` dict in the outputs.
        """
        raise NotImplementedError
