import hydra
from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()
import pytorch_lightning as pl
from omegaconf import DictConfig
from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.vis.rich_logger import print_cfg
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def test(cfg: DictConfig) -> None:
    """Train/Test"""
    Log.info(f"[Exp Name]: {cfg.exp_name}")
    if cfg.task == "fit":
        Log.info(f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}")
    pl.seed_everything(cfg.seed)

    # preparation
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    dataloader = datamodule.train_dataloader()
    for batch in tqdm(dataloader):
        for name,x in batch.items():
            if name not in ['meta', 'B']:
                if x is None:
                    raise RuntimeError(f"{name} is None: {x}")
                if type(x) is dict:
                    for k,v in x.items():
                        if not torch.isfinite(v).all():
                            raise RuntimeError(f"{name}.{k} is not finite: {v}")
                elif not torch.isfinite(x).all():
                    raise RuntimeError(f"{name} is not finite: {x}")


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    test(cfg)


if __name__ == "__main__":
    register_store_gvhmr()
    main()
