from src.simulation.cycle_updates import make_step
from src.utils.setup import initialize_n_nforks, initial_tree
from matplotlib import pyplot as plt
import numpy as np

def log_state(history, **kwargs):
    for key, value in kwargs.items():
        history[key].append(value)

def run_simulation(cfg):
    """
        These simulations returns the values of the main quantities of interest (such as volume, no of sites, 
        no of DnaA-ATP proteins, no of origins etc.) as a function of time. 
    """
    simulation_data={
        "time" : [],
        "volume" : [],
        "n_tot" : [], 
        "a_atp" : [],
        "a_adp" : [],
        "c_atp" : [],
        "c_adp" : [],
        "n_forks" : [],
        "origins" : [],
        "fpr" : []
    }
    n_tot, n_forks = initialize_n_nforks(cfg)
    tree_manager=initial_tree(cfg)
    tree_manager.n_forks=n_forks
    y=cfg.model.COOP
    t_max=cfg.simulation.T_MAX
    dt=cfg.simulation.DT
    volume, alpha, =1., 0.999
    a_atp, a_adp = alpha*cfg.model.DNAA_CONCENTRATION, (1.-alpha)*cfg.model.DNAA_CONCENTRATION
    time=0.
    count=1
    while time<t_max:
        time, a_atp, a_adp, c_atp, c_adp, volume, n_tot, f_rate = make_step(n_forks, n_tot, volume, a_atp, a_adp, time, 
                                                                          dt, y, cfg=cfg)
        tree_manager.update(time, f_rate, volume, n_tot)
        tree_manager.simulate_step(cfg)
        n_tot = tree_manager.n_tot
        volume = tree_manager.volume
        n_forks = tree_manager.n_forks
        log_state(simulation_data,
                  time=time, 
                  volume=volume, 
                  n_tot=n_tot, 
                  a_atp=a_atp,
                  a_adp=a_adp, 
                  c_atp=c_atp, 
                  c_adp=c_adp, 
                  fpr=f_rate,
                  origins=len(tree_manager.origins.keys()),
                  n_forks=n_forks
        )
        count+=1
    return simulation_data

def initiation_and_division(initiation_times, division_times, ax, cfg):
    """
        This is just needed for the plots. 
    """
    for in_time in initiation_times:
        ax.axvspan(in_time - cfg.model.LICENSING,  # left boundary
                    in_time,                          # right boundary
                    color='k',                        # fill color (black)
                    alpha=0.2,                        # transparency
                    linewidth=0)                      # no edge line

    for div_time in division_times:
        ax.axvline(div_time,linestyle='--', color='k')
