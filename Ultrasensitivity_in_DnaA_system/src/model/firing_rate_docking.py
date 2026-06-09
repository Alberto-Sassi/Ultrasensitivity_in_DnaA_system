import numpy as np
from src.utils.helpers import get_Hill
from matplotlib import pyplot as plt 
from src.utils.helpers import get_c
from src.utils.helpers import figure_params
from matplotlib.patches import FancyArrowPatch


def get_partition(xt, xd, y, ori_sites, x_dock, x_dock_adp):
    """
        Finds the partition function for the occupation of the origin region in the open conformation. 
    """
    D = (xd + xt*y + 1)**2.-4.*xt*(y-1.)
    if D<0:
        print("sqrt of negative value")
    sqrt_D = np.sqrt(D)
    lambda_plus = (xd + xt*y + 1)/2 + sqrt_D/2
    lambda_minus = (xd + xt*y + 1)/2 - sqrt_D/2
    partition_function= (-(lambda_minus**(ori_sites+1)) * (1 - lambda_plus+x_dock+x_dock_adp)+  
                         (lambda_plus**(ori_sites+1)) * (1 - lambda_minus+x_dock+x_dock_adp)) / (lambda_plus - lambda_minus)
    return partition_function

def open_probability(q_open, q_closed, epsilon_cost):
    """
        based on the partition function and on the energetic cost of melting the double strands, 
        it calculates the probability of the 'open' (initiation prone) conformation. 
    """
    log_open=np.log(q_open)-np.log(q_closed)-epsilon_cost
    phi = np.exp(log_open)
    return phi/(phi + 1)

def get_p_open(a_atp, a_adp, c_atp, c_adp, y, kori, ori_sites, epsilon_cost, kdock=1., kdockclosed=1.):
    """
        calculates the probability of open conformation starting from the concentrations of 
        proteins and occupied binding sites.  
    """
    x_dock = (a_atp-c_atp)/kdock
    x_dock_adp = (a_adp-c_adp)/kdock
    x_t_open = (a_atp-c_atp)/kori
    x_d_open = (a_adp-c_adp)/kori
    q_open = get_partition(x_t_open, x_d_open, y, ori_sites, x_dock, x_dock_adp)
    x_dock = (a_atp-c_atp)/kdockclosed
    x_dock_adp = (a_adp-c_adp)/kdockclosed
    q_closed = 1.+ x_dock + x_dock_adp
    p_open = open_probability(q_open, q_closed, epsilon_cost)
    return p_open

def get_firing_rate(p_open, k_max):
    """
        calculates the firing rate from the probability of open conformation. 
    """
    return k_max*p_open

def fr(a_atp, a_adp, c_atp, c_adp, y, kori, ori_sites, epsilon_cost, k_max, kdock=1., kdockclosed=1.):
    """
        calculates the firing rate starting from the concentrations of 
        proteins and occupied binding sites.  
    """
    p_open=get_p_open(a_atp, a_adp, c_atp, c_adp, y, kori, ori_sites, epsilon_cost, kdock, kdockclosed)
    f_rate=get_firing_rate(p_open, k_max)
    return f_rate


def make_figure(
    figsize_width=6.5,
    colors=None,
    figsize_height=6.
):
    """
    Create a 1x3 figure with small panels, each showing a sigmoid curve and up to two vertical dashed lines.
    The figure uses a golden-ratio aspect and has no axis tick labels. Common axis labels are added:
    x -> 'volume (V)', y -> 'firing rate (k)'.

    Parameters
    ----------
    panels : list of dict
        Each dict describes one panel and must have:
            - 'x' : 1D array-like, x-values (volume)
            - 'y' : 1D array-like, y-values (firing rate)
        Optional keys:
            - 'vlines' : sequence (up to length 2) with x-positions for vertical lines.
                         The first will be BLACK, the second ORANGE by default.
            - 'title'  : str, optional small title above the panel.
    figsize_width : float, optional
        Total figure width in inches. Height is computed as width / φ (golden ratio).
    dpi : int, optional
        Figure DPI.
    colors : dict or None
        Custom colors. Defaults to:
            {'curve': '#1f77b4', 'v1': '#000000', 'v2': '#ff7f0e'}
    curve_lw : float
        Line width for the sigmoid curve.
    vline_lw : float
        Line width for the dashed vertical lines.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, list[matplotlib.axes.Axes])
        The created figure and the list of axes.
    """
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

    arrow_style = '-|>,head_length=0.8,head_width=0.26' 

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

def main(n_tot=975, a=1000.,  y=50., kori=5000., ori_sites=8, epsilon_cost=13., k_max=1., fs=50, pad=8, framewidth=1., lw=1):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    volumes=10**np.linspace(-0.3, 1.5, 5000)
    c_tot=n_tot/volumes
    c_values=[get_c(dnaa=a, K=1, c_tot=c_t) for c_t in c_tot]
    hills=[]
    hills_nodock=[]
    koris=10**np.linspace(1., 4., 20)
    figure_params()
    it_worked=[]
    it_worked_nodock=[]
    for kori in koris:
        fig, axes=make_figure(figsize_width=1.8, figsize_height=1.6)
        ax=axes[0]
        count=0
        linestyle=('-', '--')
        for kdock in (1., 100000.):
            probs=[fr(a_atp=a, a_adp=1e-7, c_atp=c_atp, c_adp=1e-8, y=y, kori=kori, ori_sites=ori_sites, 
                    epsilon_cost=epsilon_cost, k_max=k_max, kdock=kdock, kdockclosed=kdock) for c_atp in c_values]
            ax.plot(volumes, probs, linestyle[count], color="#1f77b4")
            ax.set_xlabel('volume ($V$)', labelpad=pad)
            ax.set_ylabel('firing rate ($k$)', labelpad=pad)
            add_axis_arrows(ax, mutation_scale=4, lw=None, color=None, offsetx=0.04, offsety=0.05)
            ax.set_ylim(-0.02, 1.05)
            adjust(ax, framewidth)
            try:
                hill, _ =get_Hill(volumes, probs, p0=(30., 1.))
                if count==0:
                    hills.append(hill)
                    it_worked.append(True)
                else:
                    hills_nodock.append(hill)
                    it_worked_nodock.append(True)
                
            except Exception as e:
                print(e)
                if count==0:
                    it_worked.append(False)
                else:
                    it_worked_nodock.append(False)

            count+=1
        fig.savefig(rf"C:\Users\Albi\Desktop\DnaA_manuscript\figures\docking_kori={kori:.2f}_y={y}.svg")
        plt.close()
        
    fig, axes=make_figure(figsize_width=1.6, figsize_height=1.5)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    ax=axes[0]
    reduced_koris=[koris[i] for i in range(len(koris)) if it_worked[i]]
    ms=6
    ax.plot(reduced_koris, hills, '-o', markerfacecolor='none', color="#1f77b4", markersize=ms)
    reduced_koris=[koris[i] for i in range(len(koris)) if it_worked_nodock[i]]
    ax.plot(reduced_koris, hills_nodock, '-o', markerfacecolor='none', color="#ff7f0e", markersize=ms)
    ax.text(4400, 90, "with dock", color="#1f77b4")
    ax.text(3400, 80, "without dock", color="#ff7f0e")
    ax.tick_params(bottom=True, left=0.2, labelbottom=True, labelleft=True, length=1)
    ax.set_xlabel(r"$K_{ori}$ (nM)")
    ax.set_ylabel(rf"Hill coefficient $h$")
    ax.set_xticks((0, 5000, 10000))
    #ax.set_xscale('log')
    fig.savefig(rf"C:\Users\Albi\Desktop\DnaA_manuscript\figures\Docking_hill_y={y}.svg")
    fig.savefig(rf"C:\Users\Albi\Desktop\DnaA_manuscript\figures\Docking_hill_y={y}.pdf")
    #plt.show()
    return 0


if __name__=="__main__":
    main()



