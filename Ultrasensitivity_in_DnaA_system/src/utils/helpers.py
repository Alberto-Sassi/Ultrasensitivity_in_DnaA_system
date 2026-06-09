from __future__ import annotations
from scipy.optimize import curve_fit
import math
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import shutil
import json
import matplotlib as mpl
from typing import Optional


def hill_function(x, H, K):
    return x**H / (K**H + x**H)


def get_Hill(x_data, y_data, p0):
    """
        computes Hill coefficient and activation threshold from the fit of x_data and 
        y_data with a hill function.
    """
    popt, _ = curve_fit(hill_function, x_data, y_data, p0=p0)
    return popt[0], popt[1]


def get_c(dnaa, K, c_tot):
    """
        compute the concentration of occupied binding sites.
    """
    c=((dnaa+K+c_tot)-math.sqrt((dnaa+K+c_tot)**2.-4.*dnaa*c_tot))/2.
    return c


def update_alpha(alpha, n_forks, volume, cfg):
    dt=cfg.simulation.DT
    k_h0=cfg.model.INTRINSIC_H
    k_hydr=cfg.model.H_RATE
    gr=cfg.model.GROWTH_RATE
    k_e=cfg.model.EXCHANGE_RATE
    a=cfg.model.DNAA_CONCENTRATION
    alpha_new=alpha+(-k_h0*alpha-k_hydr*n_forks/(a*volume)+(gr+k_e)*(1.-alpha))*dt
    result=max(( 0. , min(( alpha_new, 1. )) ))
    #print(alpha, result)
    return result


def get_alpha(n_forks, chi0, volume, regime, alpha_max):
    """
        returns the fraction of active DnaA
    """
    chi=n_forks*chi0
    if regime=="constant":
        alpha=alpha_max*(volume-chi)/volume
    elif regime=="linear":
        alpha=volume/(volume+chi)
    else:
        print("unknown regime in get_alpha_opt")
        return 0
    return max((alpha, 1e-10))


def figure_params():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],        
        "text.usetex": False,                
        "mathtext.fontset": "dejavusans",    
        "font.size": 8,                      
        "axes.labelsize": 9,                 
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def create_figure(layout='single', figsize=(4,4), xlabel=None, ylabel=None, n_stacked=2, sharex=True, sharey=False, 
                  xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Creates a figure with a proper layout.
    
    Parameters:
    - layout: 'single', 'stacked', or 'grid' (2x2)
    - figsize: size of the square figure in inches
    - xlabel: label(s) for the x-axis (must be a list if layout='grid')
    - ylabel: label(s) for the y-axis (must be a list if layout='stacked' or 'grid')
    - n_stacked: number of stacked plots (only applies to 'stacked' layout)
    - sharex: whether to share the x-axis in stacked plots
    - sharey: whether to share the y-axis in stacked plots
    - xlim: range(s) for the x-axis (must be a list of tuples if layout='grid', single tuple if layout='stacked')
    - ylim: range(s) for the y-axis (must be a list of tuples matching panel count)
    - xticks: tick values for the x-axis (must be a list of lists if layout='grid', single list if layout='stacked')
    - yticks: tick values for the y-axis (must be a list of lists matching panel count)
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    figure_params()
    fig, ax = None, None
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    if layout == 'single':
        fig, ax = plt.subplots(figsize=figsize)
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
    
    elif layout == 'stacked':
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = fig.add_gridspec(n_stacked, 1, hspace=0)
        axes = gs.subplots(sharex=True)
        if ylabel is None or not isinstance(ylabel, list) or len(ylabel) != n_stacked:
            ylabel = [''] * n_stacked
        
        for i, ax in enumerate(axes):
            ax.set_ylabel(ylabel[i])
            if xlim:
                ax.set_xlim(xlim)
            if ylim and isinstance(ylim, list) and len(ylim) == n_stacked:
                ax.set_ylim(ylim[i])
            if xticks:
                ax.set_xticks(xticks)
            if yticks and isinstance(yticks, list) and len(yticks) == n_stacked:
                ax.set_yticks(yticks[i])
        
        axes[-1].set_xlabel(xlabel)
        ax = axes

    elif layout == 'grid':
        fig = plt.figure(figsize=figsize)
        panel_positions = [
            [0.13, 0.58, 0.32, 0.32],  # Top-left
            [0.58, 0.58, 0.32, 0.32],  # Top-right
            [0.13, 0.13, 0.32, 0.32],  # Bottom-left
            [0.58, 0.13, 0.32, 0.32],  # Bottom-right
        ]
        
        axes = []
        for pos in panel_positions:
            ax = fig.add_axes(pos)
            axes.append(ax)
        
        if xlabel:
            axes[2].set_xlabel(xlabel[2])
            axes[3].set_xlabel(xlabel[3])
            axes[0].set_xlabel(xlabel[0])
            axes[1].set_xlabel(xlabel[1])
        if ylabel:
            axes[0].set_ylabel(ylabel[0])
            axes[1].set_ylabel(ylabel[1])
            axes[2].set_ylabel(ylabel[2])
            axes[3].set_ylabel(ylabel[3])
        
        ax=np.array(axes).reshape(2, 2)
    
    else:
        raise ValueError("Invalid layout. Choose 'single', 'stacked', or 'grid'.")
    
    return fig, ax


def make_palette(n, cmap_name="viridis", as_hex=True):
    """
    Return a list of n colors sampled from a Matplotlib colormap.

    Parameters
    ----------
    n : int
        Number of colors to generate (must be >= 1).
    cmap_name : str, optional
        Name of the colormap (e.g., "viridis", "plasma", "cividis", "magma", "tab10").
        See available names in matplotlib.colormaps().
    as_hex : bool, optional
        If True, return colors as hex strings (e.g., "#4c78a8").
        If False, return RGBA tuples.

    Returns
    -------
    list
        A list of colors (hex strings or RGBA tuples), length n.

    Raises
    ------
    ValueError
        If n < 1 or colormap name is invalid.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("`n` must be a positive integer (>= 1).")

    try:
        import numpy as np
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError as e:
        raise ImportError("This function requires Matplotlib and NumPy.") from e

    # Get the colormap; will raise if name is invalid
    try:
        cmap = cm.get_cmap(cmap_name)
    except ValueError as e:
        # Suggest available colormaps in the error to help debugging
        available = list(cm.cmap_d.keys()) if hasattr(cm, "cmap_d") else list(cm.colormaps())
        raise ValueError(f"Unknown colormap '{cmap_name}'. Try one of: {available[:10]} ...") from e

    # Sample evenly spaced points across the colormap, including endpoints
    positions = np.linspace(0.0, 1.0, n)
    rgba_colors = [cmap(p) for p in positions]

    if as_hex:
        return [mcolors.to_hex(c) for c in rgba_colors]
    else:
        return rgba_colors
    

def format_param_value(value) -> str:
    """
    Format a parameter value for use in a folder name:
    - Floats: replace '.' with 'p' (e.g., 3.0 -> '3p0', 0.25 -> '0p25')
    - Ints: unchanged (e.g., 5 -> '5')
    - Strings: passed through, but any '.' are replaced by 'p' to be consistent.
    """
    if isinstance(value, float):
        # Use plain string conversion without scientific notation for typical values
        s = f"{value:.15g}"  # avoids trailing .0 when unnecessary, but keeps decimal if present
        s = s.replace(".", "p")
        return s
    elif isinstance(value, int):
        return str(value)
    else:
        # Fallback: treat as string and normalize dots
        return str(value).replace(".", "p")


def strip_yaml_suffix(name: str) -> str:
    """
    Remove a trailing '.yaml' from the given name if present.
    Keeps other dots intact (e.g., 'config.v2.yaml' -> 'config.v2').
    """
    return name[:-5] if name.endswith(".yaml") else name


def build_output_path(
    base_name: str,
    sweep_name: Optional[str] = None,
    root: Path | str = "results",
    params: Path | str = "runs",
    filename: str = "cv.json",
    filetype: Path | str = "data",
) -> Path:
    """
    Build the output path. If sweep_name is provided:
      results/base=<basename>__sweep=<sweepname>/<params>/<filetype>/<filename>
    Else:
      results/base=<basename>/<params>/<filetype>/<filename>

    """
    base_clean = strip_yaml_suffix(base_name)
    sweep_clean = strip_yaml_suffix(sweep_name) if sweep_name is not None else None
    if sweep_clean is None:
        dir_part = f"base={base_clean}"
    else:
        dir_part = f"base={base_clean}__sweep={sweep_clean}"

    root = Path(root)
    params = Path(params)
    filetype = Path(filetype)
    dir_path = root / dir_part / params / filetype
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path / filename


def float_to_string(a, fmt=".3f"):
    return format(a, fmt).replace('.', 'p')


def filename_from_path(path):
    return Path(path).name


def copy_config_files(base_path, sweep_path=False, root_dir="results"):
    """
    Create a directory named 'base=<base_name>__sweep=<sweep_name>' and copy the two YAML files into it.

    - base_name and sweep_name are derived from the file names without the '.yaml' suffix.
    - The directory is created if it doesn't already exist.
    - Files are copied with metadata preserved.

    Parameters
    ----------
    base_path : str or Path
        Path to the base YAML file (e.g., 'src/config/base1.yaml').
    sweep_path : str or Path
        Path to the sweep YAML file (e.g., 'src/config/sweepA.yaml').
    root_dir : str or Path, optional
        Directory under which the target folder will be created. Default is current directory (".")

    Returns
    -------
    Path
        The Path to the created (or existing) target directory.

    Raises
    ------
    FileNotFoundError
        If either input path does not exist.
    ValueError
        If either file does not have a '.yaml' (or '.yml') suffix.
    """
    base_path = Path(base_path)
    if sweep_path:
        sweep_path = Path(sweep_path)
    root_dir = Path(root_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base YAML not found: {base_path}")
    
    if sweep_path:
        if not sweep_path.exists():
            raise FileNotFoundError(f"Sweep YAML not found: {sweep_path}")

    def _ensure_yaml(p: Path):
        if p.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Expected a YAML file, got: {p.name}")

    _ensure_yaml(base_path)
    if sweep_path:
        _ensure_yaml(sweep_path)

    base_name = base_path.stem
    if sweep_path:
        sweep_name = sweep_path.stem
    else:
        sweep_name="na"
    target_dir_name = f"base={base_name}__sweep={sweep_name}"
    target_dir = root_dir / target_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_base = target_dir / base_path.name
    shutil.copy2(base_path, dest_base)
    if sweep_path:
        dest_sweep = target_dir / sweep_path.name
        shutil.copy2(sweep_path, dest_sweep)

    return target_dir


def make_some_plots(kori, y, base_name, sweep_name):
    y_string=float_to_string(y)
    kori_string=float_to_string(kori)
    file_path= build_output_path(base_name=base_name, sweep_name=sweep_name, 
                                    filetype="data", params=rf"y={y_string}__kori={kori_string}", filename="traces.json")

    with open(file_path, "r") as f:
        simulation_data=json.load(f)

    time=simulation_data["time"]
    volume=simulation_data["volume"]
    origins=simulation_data["origins"]
    n_tot=simulation_data["n_tot"]

    a_atp=np.array(simulation_data["a_atp"])
    a_adp=np.array(simulation_data["a_adp"])


    plt.figure("volume")
    plt.plot(time, volume)
    plt.figure("origins")
    plt.plot(time, origins)
    plt.figure("alpha")
    plt.plot(time, a_atp/(a_atp+a_adp))
    plt.figure("n tot")
    plt.plot(time, n_tot)            
    plt.show()