import numpy as np
from src.simulation.fork_tracker import TreeManager
from src.utils.helpers import get_alpha

def term_init_cycles(cfg):
    """
        The function returns the time of division, initiation, termination, and the largest
        number of co-existing replication cycles. 
    """
    tau=np.log(2.)/cfg.model.GROWTH_RATE
    t_ter = tau - cfg.model.D
    cycles=np.floor((cfg.model.REP_TIME+cfg.model.D)/tau)
    t_in = (cycles+1)*tau - cfg.model.REP_TIME - cfg.model.D
    return tau, t_in, t_ter, cycles

def n_forks_at_initiation(t_in, t_ter, cycles):
    """
        The function returns the total number of ongoing replication forks. 
    """
    n_origins=int(2**cycles)
    if t_in<t_ter:
        n_forks=int(2*(n_origins-1))
    else:
        n_forks=int(4*(n_origins/2-1))
    return n_forks     

def n_sites_at_initiation(cfg, tau, t_in, t_ter):
    """
        Given the division time, the time at termination and the time at 
        initiation, the function calculates the total number of sites at the moment of initiation. 
    """
    rounds = int(cfg.model.REP_TIME // tau)
    sites_at_init = cfg.model.SITES*(1 + sum(((rounds - i) * tau / cfg.model.REP_TIME) * (2 ** i) 
                                              for i in range(rounds)))
    if t_in>t_ter:
        return 2*sites_at_init
    else:
        return sites_at_init   

def sites_concentration(n_sites, volume):
    return n_sites/volume

def initial_tree(cfg):
    """
        The tree is needed to keep track of all the origins. 
    """
    tree_manager = TreeManager(cfg)
    tree_manager.add_initial_origin()
    return tree_manager

def get_chi(cfg, v_opt):
    """
        chi0=n_forks*chi is a parameter quantifying the activation/deactivation dynamics. 
        It has a different expression depending on the approximation of Michaelis 
        Menten we are using for the RIDA system. 
    """
    return v_opt/cfg.model.CHANGE

def get_alpha_opt(cfg, v_opt):
    regime=cfg.model.REGIME
    chi=get_chi(cfg, v_opt)
    chi0=chi/n_forks
    _, n_forks = get_n_star_n_forks(cfg)
    if regime=="constant":
        alpha_opt=get_alpha(n_forks, chi0=chi0, volume=v_opt, regime=regime)
    elif regime=="linear":
        alpha_opt=v_opt/(v_opt+chi0)
    else:
        print("unknown regime in get_alpha_opt: %s"%regime)
        return 0
    return max((alpha_opt, 1e-10))

def get_n_star_n_forks(cfg):
    """
        Finds the number of titration sites and the number of forks at the moment of initiation. 
    """
    tau, t_in, t_ter, cycles = term_init_cycles(cfg)
    sites_at_init = n_sites_at_initiation(cfg, tau, t_in, t_ter)
    forks_at_init = n_forks_at_initiation(t_in, t_ter, cycles)
    return sites_at_init, forks_at_init

def get_v_opt(cfg, n_star):
    """
        The optimal volume is the volume at midpoint when the parameters are chosen to maximize the 
        Hill coefficient. 
    """
    v_opt=n_star/cfg.model.DNAA_CONCENTRATION
    return v_opt
   
def get_y_opt(cfg, alpha_opt):
    """
        For a given dissociation constant Kori, we find the cooperativity strength that maximizes 
        the Hill coefficient. 
    """
    kori, epsilon_cost, origin_sites, K, dnaa=(cfg.model.K_OPEN,
                                               cfg.model.E_COST,
                                               cfg.model.ORIGIN_SITES,
                                               cfg.model.K,
                                               cfg.model.DNAA_CONCENTRATION)
    y_opt=kori*np.exp(epsilon_cost/origin_sites)/(alpha_opt*np.sqrt(K*dnaa))
    return y_opt

def initialize_chi0(cfg):
    n_star, _=get_n_star_n_forks(cfg)
    v_opt=get_v_opt(cfg, n_star)
    chi=get_chi(cfg, v_opt)
    return chi

def initialize_y(cfg):
    """
        The way the parameters are selected is the following: 
        - calculation of the number of titration sites at the moment of initiation. 
        - calculation of the corresponding optimal volume at initiation. 
        - a chi0 is selected, based on the optimal volume. 
        - calculation of the optimal alpha based on V_opt and chi.
        - calculation of the optimal cooperativity strength.  
    """
    n_star, _=get_n_star_n_forks(cfg)
    v_opt=get_v_opt(cfg, n_star)
    alpha_opt=get_alpha_opt(cfg, v_opt)
    y_opt=get_y_opt(cfg, alpha_opt)
    return y_opt

def initialize_n_nforks(cfg):
    """
        At the beginning of the simulations there is a single genome and no ongoing replication. 
        If the conditions or stability are met, after a certain relaxation time the system 
        converges to the steady state with the proper replication profile. 
    """
    n_forks = 0.
    n_tot = cfg.model.SITES
    return n_tot, n_forks
