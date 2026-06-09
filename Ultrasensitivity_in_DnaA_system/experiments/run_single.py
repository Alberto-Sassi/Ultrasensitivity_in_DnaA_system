import argparse
from src.utils.config_loader import load_config
from src.simulation.run_simulation import run_simulation
from src.utils.helpers import copy_config_files, filename_from_path, build_output_path
import json
import os


def load_parameters():
    parser = argparse.ArgumentParser(
        description="Run one simulation from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML_FILE",
        help="Path to the YAML config (e.g., configs/base.yaml)"
    )
    args = parser.parse_args()
    base_path=args.config
    cfg = load_config(base_path)
    base_name=filename_from_path(base_path)
    print("kori, y = ", cfg.model.K_OPEN, cfg.model.COOP)
    return cfg, base_name


def simulate_and_save(cfg, base_name):
    """
    Runs one simulation for (y, kori) and writes the JSON output to the appropriate path.
    Returns (y, kori, file_path) on success.
    """
    simulation_data = run_simulation(cfg)
    file_path = build_output_path(
        base_name=base_name, 
        filetype="data",
        params=rf"",
        filename="traces.json"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(simulation_data, f, indent=2)

    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Run one simulation from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML_FILE",
        help="Path to the YAML config (e.g., configs/base.yaml)"
    )
    args = parser.parse_args()
    cfg, base_name = load_parameters()
    file_path = simulate_and_save(cfg, base_name)

    print("Loaded config:", args.config)
    print("file path =", file_path)


if __name__ == "__main__":
    main()