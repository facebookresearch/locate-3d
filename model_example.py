from models.locate_3d import Locate3D
from omegaconf import OmegaConf

cfg = OmegaConf.load("config/locate_3d.yaml")

model = Locate3D(cfg)

model.load_from_checkpoint("/fsx-cortex/sergioarnaud/accel-cortex-m2f-sergioarnaud/2025-01-23_07-03-42_lam3d_train_jan20-locatex/0/checkpoints/ema_epoch_34.pt")

breakpoint()
