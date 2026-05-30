import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        tta_adapter=None,
        writer=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            tta_adapter (BaseTTA): test-time adaptation strategy.
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
        """
        assert tta_adapter is not None, "Provide initialized TTA adapter"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms
        self.tta_adapter = tta_adapter
        self.writer = writer

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=self.writer,
            )
        else:
            self.evaluation_metrics = None

        pretrained_path = config.inferencer.get("from_pretrained")
        if pretrained_path is not None:
            self._from_pretrained(pretrained_path)

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    @staticmethod
    def _scalar_logs(logs):
        if logs is None:
            return None
        result = {}
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            result[key] = value
        return result

    def _log_scalars_to_writer(self, scalars, prefix):
        if self.writer is None or not scalars:
            return
        self.writer.add_scalars(
            {f"{prefix}/{key}": value for key, value in scalars.items()}
        )

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and optionally
        save predictions to disk.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.tta_adapter.predict_batch(batch)
        batch.update(outputs)
        self._log_scalars_to_writer(self._scalar_logs(batch.get("tta_logs")), "tta")

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))
            self._log_scalars_to_writer(metrics.result(), "metrics")

        if self.cfg_trainer.get("save_predictions", False):
            self._save_predictions(batch_idx, batch, part)

        return batch

    def _save_predictions(self, batch_idx, batch, part):
        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)

            output = {
                "pred_label": pred_label,
                "label": label,
            }

            if self.save_path is not None:
                output_id = current_id + i
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()
        if self.writer is not None:
            self.writer.set_step(0, part)

        prep_logs = self.tta_adapter.before_partition(
            dataloader=dataloader,
            move_batch_to_device=self.move_batch_to_device,
            transform_batch=self.transform_batch,
        )
        self._log_scalars_to_writer(self._scalar_logs(prep_logs), "tta")

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc=part,
            total=len(dataloader),
        ):
            if self.writer is not None:
                self.writer.set_step(batch_idx, part)
            batch = self.process_batch(
                batch_idx=batch_idx,
                batch=batch,
                part=part,
                metrics=self.evaluation_metrics,
            )

        logs = self.evaluation_metrics.result()
        if self.writer is not None:
            self.writer.set_step(len(dataloader), part)
            self._log_scalars_to_writer(logs, "final_metrics")

        return logs
