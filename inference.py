import warnings
import logging

import hydra
import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="inference_gptsniffer"
)
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)
    logger = logging.getLogger("inference")

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)
    tta_adapter = instantiate(config.tta, model=model)

    # get metrics
    metrics = instantiate(config.metrics)
    writer = None
    if config.get("writer") is not None:
        project_config = OmegaConf.to_container(config, resolve=True)
        writer_config = OmegaConf.to_container(config.writer, resolve=True)
        writer_cls = get_class(writer_config.pop("_target_"))
        writer = writer_cls(
            logger=logger,
            project_config=project_config,
            **writer_config,
        )

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        tta_adapter=tta_adapter,
        writer=writer,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")

    if writer is not None:
        writer.finish()


if __name__ == "__main__":
    main()
