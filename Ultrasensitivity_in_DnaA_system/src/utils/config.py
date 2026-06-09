
# src/utils/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelParams:
    SITES: int
    REP_TIME: float
    GROWTH_RATE: float
    ORIGIN_SITES: float
    E_COST: float
    FIRING_MAX: float
    K_OPEN: float
    D: float
    ECLIPSE: float
    LICENSING: float
    DNAA_CONCENTRATION: float
    K: float
    CHI0: float
    COOP: float
    REGIME: str
    ALPHA_MAX: float 

@dataclass(frozen=True)
class SimulationParams:
    seed: int
    T_MAX: float
    DT: float

@dataclass(frozen=True)
class Config:
    model: ModelParams
    simulation: SimulationParams
    output_dir: str
    dpi: int
