import numpy as np

def get_partition(xt, xd, y, ori_sites):
    """
        Finds the partition function for the occupation of the origin region in the open conformation. 
    """
    D = (xd + xt*y + 1)**2.-4.*xt*(y-1.)
    if D<0:
        print("sqrt of negative value")
    sqrt_D = np.sqrt(D)
    lambda_plus = (xd + xt*y + 1)/2 + sqrt_D/2
    lambda_minus = (xd + xt*y + 1)/2 - sqrt_D/2
    return lambda_plus**ori_sites + lambda_minus**ori_sites

def open_probability(q_open, epsilon_cost):
    """
        based on the partition function and on the energetic cost of melting the double strands, 
        it calculates the probability of the 'open' (initiation prone) conformation. 
    """
    log_open=np.log(q_open)-epsilon_cost
    open_term = np.exp(log_open)
    return open_term/(open_term + 1)

def get_p_open(a_atp, a_adp, c_atp, c_adp, y, kori, ori_sites, epsilon_cost):
    """
        calculates the probability of open conformation starting from the concentrations of 
        proteins and occupied binding sites.  
    """
    x_t_open = (a_atp-c_atp)/kori
    x_d_open = (a_adp-c_adp)/kori
    q_open = get_partition(x_t_open, x_d_open, y, ori_sites)
    p_open = open_probability(q_open, epsilon_cost)
    return p_open

def get_firing_rate(p_open, k_max):
    """
        calculates the firing rate from the probability of open conformation. 
    """
    return k_max*p_open

def fr(a_atp, a_adp, c_atp, c_adp, y, kori, ori_sites, epsilon_cost, k_max):
    """
        calculates the firing rate starting from the concentrations of 
        proteins and occupied binding sites.  
    """
    p_open=get_p_open(a_atp, a_adp, c_atp, c_adp, y, kori, ori_sites, epsilon_cost)
    f_rate=get_firing_rate(p_open, k_max)
    return f_rate


from matplotlib import pyplot as plt 
from src.utils.helpers import get_c
from src.utils.helpers import figure_params
import matplotlib.pyplot as plt


def make_figure(figsize_width=6.5, colors=None, figsize_height=6.):
    figsize = (figsize_width, figsize_height)
    if colors is None:
        colors = {'curve': '#1f77b4', 'v1': '#000000', 'v2': '#ff7f0e'}

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axes = []
    ax = fig.add_subplot(111)
    axes.append(ax)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False, length=0)
    ax.margins(x=0.05, y=0.05)
    return fig, axes

def adjust(ax, framewidth):
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(framewidth)

    for spine in ('right', 'top'):
        ax.spines[spine].set_visible(False)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def add_axis_arrows(ax, *, mutation_scale=20, lw=None, color=None, offsetx=0.0015, offsety=0.015):
    """
    Add arrowheads to the positive ends of x and y axes using axes coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    mutation_scale : float
        Size of the arrowhead.
    lw : float or None
        Line width of the arrow; defaults to axes.linewidth.
    color : str or tuple or None
        Color of the arrow; defaults to spine color (left/bottom).
    offset : float
        How far beyond [0,1] in axes coords the arrow tip extends.
    """
    if lw is None:
        lw = plt.rcParams.get('axes.linewidth', 1.0)
    if color is None:
        color = ax.spines['left'].get_edgecolor()

    arrow_style = '-|>,head_length=0.8,head_width=0.26'  # triangular clean head
    x_arrow = FancyArrowPatch(
        (0.0, 0.0), (1.0 + offsetx, 0.0),
        transform=ax.transAxes,
        arrowstyle=arrow_style,
        mutation_scale=mutation_scale,
        lw=lw, color=color,
        clip_on=False, zorder=5,
        shrinkA=0.0, shrinkB=0.0,
    )
    ax.add_patch(x_arrow)
    y_arrow = FancyArrowPatch(
        (0.0, 0.0), (0.0, 1.0 + offsety),
        transform=ax.transAxes,
        arrowstyle=arrow_style,
        mutation_scale=mutation_scale,
        lw=lw, color=color,
        clip_on=False, zorder=5,
        shrinkA=0.0, shrinkB=0.0,
    )
    ax.add_patch(y_arrow)


def main(n_tot=975, a=1000.,  y=50, kori=5000., ori_sites=8, epsilon_cost=13., k_max=1., fs=50, pad=8, framewidth=1., lw=1):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    volumes=10**np.linspace(-0.3, 0.4, 1000)
    c_tot=n_tot/volumes
    c_values=[get_c(dnaa=a, K=1, c_tot=c_t) for c_t in c_tot]
    figure_params()
    for kori in (20., 300., 5000.):
        fig, axes=make_figure(figsize_width=1.4, figsize_height=1.4)
        ax=axes[0]
        probs=[fr(a_atp=a, a_adp=1e-7, c_atp=c_atp, c_adp=1e-8, y=y, kori=kori, ori_sites=ori_sites, epsilon_cost=epsilon_cost, k_max=k_max) for c_atp in c_values]
        midpoint=np.argmin((np.array(probs)-0.5)**2.)
        ax.axvline(np.log(n_tot/a), linestyle='--', color='k')
        ax.axvline(np.log(volumes[midpoint]), linestyle='--', color='#ff7f0e')
        ax.text(np.log(n_tot/a)+0.05, 0.5, r"$\frac{n^*}{a}$", color='k')
        ax.text(np.log(volumes[midpoint])-0.3, 0.5, r"$V^*$", color='#ff7f0e')
        ax.plot(np.log(volumes), probs, color="#1f77b4")
        ax.set_xlabel('volume ($V$)', labelpad=pad)
        ax.set_ylabel('firing rate ($k$)', labelpad=pad)
        add_axis_arrows(ax, mutation_scale=4, lw=None, color=None, offsetx=0.04, offsety=0.05)
        ax.set_ylim(-0.02, 1.05)
        adjust(ax, framewidth)
        fig.savefig(rf"C:\Users\Albi\Desktop\DnaA_manuscript\figures\{kori:.2f}.svg")
    return 0


if __name__=="__main__":
    main()



