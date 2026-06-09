
import argparse
from src.utils.config_loader import load_config
from src.simulation.run_simulation import run_simulation
import json
from dataclasses import replace
import yaml
import numpy as np
from src.utils.helpers import build_output_path, float_to_string, copy_config_files, filename_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import os
import json

def get_range(sweep_dict):
    max_v=sweep_dict["max_val"]
    min_v=sweep_dict["min_val"]
    steps=sweep_dict["steps"]
    values=np.exp(np.linspace(min_v, max_v, steps))    
    
    return values


def load_parameters():
    """
    load the parameters from a yaml file. The folder of the yaml files is src/configs.
    """
    parser = argparse.ArgumentParser(
        description="Run one simulation from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML_FILE",
        help="Path to the YAML config (e.g., configs/base.yaml)"
    )
    parser.add_argument(
        "--sweep",
        required=True,
        metavar="YAML_FILE",
        help="Path to the YAML sweep"
    )
    args = parser.parse_args()
    sweep_path = args.sweep
    sweep_dict = yaml.safe_load(open(sweep_path))
    params=sweep_dict["params"]
    sweep_coop=params["COOP"]
    sweep_kori=params["K_OPEN"]
    simul=sweep_dict["simulation"]
    t_max=simul["T_MAX"]
    dt=simul["DT"]
    base_path=args.config
    copy_config_files(base_path, sweep_path)
    cfg = load_config(base_path)
    cfg = replace(cfg, simulation=replace(cfg.simulation, T_MAX=t_max))
    cfg = replace(cfg, simulation=replace(cfg.simulation, DT=dt))
    y_values=get_range(sweep_coop)
    kori_values=get_range(sweep_kori)
    base_name = filename_from_path(base_path)
    sweep_name = filename_from_path(sweep_path)
    return cfg, y_values, kori_values, base_name, sweep_name


def simulate_and_save(y, kori, cfg, base_name, sweep_name):
    """
    Runs one simulation and writes the JSON output to the appropriate path.
    Returns (y, kori, file_path) on success.
    """
    cfg0 = replace(cfg, model=replace(cfg.model, COOP=y, K_OPEN=kori))
    simulation_data = run_simulation(cfg0)
    y_string = float_to_string(y)
    kori_string = float_to_string(kori)
    file_path = build_output_path(
        base_name=base_name, 
        sweep_name=sweep_name,
        filetype="data",
        params=rf"y={y_string}__kori={kori_string}",
        filename="traces.json"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(simulation_data, f, indent=2)

    return (y, kori, file_path)


def main():
    cfg, y_values, kori_values, base_name, sweep_name = load_parameters()
    combos = list(product(kori_values, y_values))
    max_workers = os.cpu_count() or 4
    futures = []
    completed = 0
    total = len(combos)

    print(f"Submitting {total} simulations with up to {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for kori, y in combos:
            fut = ex.submit(simulate_and_save, y, kori, cfg, base_name, sweep_name)
            futures.append(fut)

        for fut in as_completed(futures):
            try:
                y, kori, file_path = fut.result()
                completed += 1
                print(f"[{completed}/{total}] Done: y={y}, kori={kori} → {file_path}")
            except Exception as e:
                print("A simulation failed with error:", repr(e))

    print("All submitted simulations complete.")


if __name__ == "__main__":
    main()
