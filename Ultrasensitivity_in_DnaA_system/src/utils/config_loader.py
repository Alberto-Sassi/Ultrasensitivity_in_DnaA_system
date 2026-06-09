
# src/utils/config_loader.py
import yaml
from .config import Config, ModelParams, SimulationParams

def load_config(path: str) -> Config:
    data = yaml.safe_load(open(path))
    mp = ModelParams(**data["model"])
    sp = SimulationParams(**data.get("simulation", {}))
    out_dir = data["output"]["dir"]
    dpi = data["output"]["dpi"]
    return Config(model=mp, simulation=sp, output_dir=out_dir, dpi=dpi)
