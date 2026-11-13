"""Stability argument for finite trajectory averages of sPHNNs (methodology chapter, Figure 3.1; illustrates Equation (3.8) and Corollary A.4).

Two linear port-Hamiltonian members, x' = (j_k J0 - rho_k I) P_k x with H_k(x) = x^T P_k x / 2 and the equilibrium x* at the origin, start at the same state x0 and are averaged with equal weights.
  epsilon_H   = min_k lambda_min(P_k) / 2          common constant with H_k(x) >= epsilon_H |x - x*|^2
  bound set   {x : Hbar(x) < epsilon_H r_stab^2}   Hbar = mean of the H_k; the initial states for which Equation (3.8) keeps the mean within r_stab (a condition on x0, not a region the mean is confined to)
  delta       radius of a disc around x* inside the bound set; the largest such disc touches the boundary and is drawn at 85 % of its radius
The asserts check delta disc < bound set < r_stab disc and the bound of Equation (3.8) along the computed mean.
Output: the export folder (imgs/main/method/stability_of_averages), drawn at the inclusion width of figure_widths.json. Usage: python make_stability_schematic.py"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import export_paths as EP
import plot_lib as pl                                 # thesis plot style (pl.tps, pl.style)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon

tps, style = pl.tps, pl.style
KEY = "stability_of_averages"
EXPORT = EP.KB / "imgs" / "main" / "method" / KEY

# ---- members and initial state ----
P1 = np.array([[2.7356, 2.2547], [2.2547, 4.0189]])
P2 = np.array([[1.2574, -0.3328], [-0.3328, 3.0034]])
J_GAIN, RHO = (0.6861, 0.8143), (0.07, 0.13)          # J_k = j_k [[0, -1], [1, 0]], R_k = rho_k I
X0 = np.array([-0.342, 0.198]) * (0.30 / np.hypot(-0.342, 0.198))
T_END, N_STEPS = 15.0, 1500
J0 = np.array([[0.0, -1.0], [1.0, 0.0]])
DELTA_FRACTION = 0.85

# ---- layout: plot area in inches, in units of r_stab, label positions in degrees ----
AXES_IN, LIM, LEGEND_BAND_IN, PAD_IN = 2.47, 1.16, 0.32, 0.05
DELTA_LABEL_DEG, RSTAB_LABEL_DEG = -30.0, 45.0
MS_EQUILIBRIUM = 3.4

COL_DELTA, COL_BOUND, COL_RSTAB = style.ROLE_C, style.ROLE_A, "0.45"
mix = lambda c, a: tuple(a * v + (1 - a) * 1.0 for v in to_rgb(c))      # colour c at opacity a over white (the same fill in the plot and in the legend)
FILL_RSTAB, FILL_BOUND, FILL_DELTA = mix(COL_RSTAB, 0.12), mix(COL_BOUND, 0.18), mix(COL_DELTA, 0.30)
BOUND_LS = (0, (4, 2.5))
LW_SET, LW_MEMBER, LW_MEAN = style.LW["reference"], 0.9, style.LW["mean"] + 0.3


def simulate(P, j, rho, x0):
    """The linear system on a uniform time grid (matrix exponential from the eigendecomposition of the 2 x 2 system matrix)."""
    w, V = np.linalg.eig((j * J0 - rho * np.eye(2)) @ P)
    step = (V @ np.diag(np.exp(w * T_END / (N_STEPS - 1))) @ np.linalg.inv(V)).real
    X = np.zeros((N_STEPS, 2))
    X[0] = x0
    for i in range(1, N_STEPS):
        X[i] = step @ X[i - 1]
    return X


def build():
    """Members, mean and the three nested sets; asserts the relations of the argument."""
    Ps, pi = (P1, P2), np.array([0.5, 0.5])
    eps, r = min(np.linalg.eigvalsh(P)[0] for P in Ps) / 2, 1.0
    Pb = sum(w * P for w, P in zip(pi, Ps))
    lam, V = np.linalg.eigh(Pb)
    semi = np.sqrt(2 * eps * r**2 / lam)                                   # semi-axes of {Hbar < eps r^2}, long axis first
    th = np.linspace(0, 2 * np.pi, 721)
    bound_xy = (V @ (semi[:, None] * np.stack([np.cos(th), np.sin(th)]))).T
    delta = DELTA_FRACTION * semi[1]
    X = [simulate(P, j, rho, X0) for P, j, rho in zip(Ps, J_GAIN, RHO)]
    M = sum(w * Xk for w, Xk in zip(pi, X))
    Hbar = lambda Z: 0.5 * np.einsum("...i,ij,...j->...", Z, Pb, Z)
    circle = np.stack([np.cos(th), np.sin(th)], axis=1)
    bound = math.sqrt(Hbar(X0) / eps)                                      # Equation (3.8): |xbar(t) - x*| <= sqrt(Hbar(x0) / eps_H)
    assert np.linalg.norm(X0) < delta
    assert Hbar(delta * circle).max() < eps * r**2
    assert np.linalg.norm(bound_xy, axis=1).max() < r
    assert all(0.5 * X0 @ P @ X0 >= eps * (X0 @ X0) for P in Ps)
    assert np.linalg.norm(M, axis=1).max() <= bound + 1e-12 and bound < r
    assert all(np.all(np.diff(0.5 * np.einsum("ni,ij,nj->n", Xk, P, Xk)) <= 1e-12) for Xk, P in zip(X, Ps))
    return dict(members=X, mean=M, r_stab=r, delta=delta, bound_xy=bound_xy)


def put_text(ax, xy, text, **kw):
    return ax.text(*xy, text, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), fontsize=tps.raw_pt("annot"), zorder=9, **kw)


def add_legend(fig, handles, labels, expand):
    kw = dict(loc="upper left", frameon=False, ncol=len(handles), handlelength=1.2, handletextpad=0.4, borderaxespad=0.0)
    if expand:
        return fig.legend(handles, labels, bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), mode="expand", **kw)         # entries spread over the figure width
    return fig.legend(handles, labels, bbox_to_anchor=(0.0, 1.0), columnspacing=0.6, **kw)


def draw(data):
    tps.apply_style(KEY)
    W = tps.sizes.width_in(KEY)
    H = AXES_IN + LEGEND_BAND_IN + PAD_IN
    r, delta, window = data["r_stab"], data["delta"], LIM * data["r_stab"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([(W - AXES_IN) / 2 / W, PAD_IN / H, AXES_IN / W, AXES_IN / H])
    ax.set_xlim(-window, window)
    ax.set_ylim(-window, window)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Circle((0, 0), r, fc=FILL_RSTAB, ec=COL_RSTAB, lw=LW_SET, zorder=1))
    ax.add_patch(Polygon(data["bound_xy"], closed=True, fc=FILL_BOUND, ec=COL_BOUND, lw=LW_SET, ls=BOUND_LS, zorder=2))
    ax.add_patch(Circle((0, 0), delta, fc=FILL_DELTA, ec=COL_DELTA, lw=LW_SET, zorder=3))
    for a, b in (((-0.97 * window, 0), (0.97 * window, 0)), ((0, -0.97 * window), (0, 0.97 * window))):      # state-space axes through x*, over the sets and under the curves
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.7, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=3.5)
    put_text(ax, (0.985 * window, -0.045 * r), r"$x_1$", ha="right", va="top")
    put_text(ax, (0.045 * r, 0.985 * window), r"$x_2$", ha="left", va="top")

    for Xk in data["members"]:
        ax.plot(Xk[:, 0], Xk[:, 1], color=style.MEMBER["color"], ls=style.MEMBER["ls"], lw=LW_MEMBER, zorder=4)
    ax.plot(data["mean"][:, 0], data["mean"][:, 1], color=style.MEAN_COLOR, lw=LW_MEAN, solid_capstyle="round", zorder=5)
    ax.plot([0], [0], "o", color=style.BLACK, ms=MS_EQUILIBRIUM, zorder=7)
    ax.plot([X0[0]], [X0[1]], "o", mfc="white", mec=style.BLACK, mew=0.8, ms=4.6, zorder=8)

    put_text(ax, X0 + np.array([0.0, 0.095 * r]), r"$x_0$")
    for radius, deg, text in ((delta, DELTA_LABEL_DEG, r"$\delta$"), (r, RSTAB_LABEL_DEG, r"$r_{\mathrm{stab}}$")):          # name the boundary line: just outside the circle
        a = math.radians(deg)
        put_text(ax, (1.04 * radius * math.cos(a), 1.04 * radius * math.sin(a)), text, ha="left" if math.cos(a) > 0 else "right", va="bottom" if math.sin(a) > 0 else "top")

    handles = [Patch(fc=FILL_DELTA, ec=COL_DELTA, lw=LW_SET), Patch(fc=FILL_BOUND, ec=COL_BOUND, lw=LW_SET, ls=BOUND_LS), Patch(fc=FILL_RSTAB, ec=COL_RSTAB, lw=LW_SET),
               Line2D([], [], color=style.MEMBER["color"], ls=style.MEMBER["ls"], lw=LW_MEMBER), Line2D([], [], color=style.MEAN_COLOR, lw=LW_MEAN),
               Line2D([], [], ls="", marker="o", color=style.BLACK, ms=MS_EQUILIBRIUM)]
    labels = [r"$\|x_0-x^*\|<\delta$", r"$\overline{\mathcal{H}}(x_0)<\epsilon_H\,r_{\mathrm{stab}}^{2}$", r"$\|\overline{x}(t)-x^*\|<r_{\mathrm{stab}}$", r"members $x_k(t)$", r"mean $\overline{x}(t)$", r"$x^*$"]
    natural = add_legend(fig, handles, labels, expand=False)
    fig.canvas.draw()
    assert natural.get_window_extent().width / fig.dpi <= W - 2 * 0.03, "the legend does not fit into one row at this width"
    natural.remove()
    add_legend(fig, handles, labels, expand=True)
    return fig


if __name__ == "__main__":
    tps.save_figure(draw(build()), EXPORT, generator="02_harmonic_oscillator/figures/make_stability_schematic.py")
