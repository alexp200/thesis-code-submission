"""Dataset introduction figures of the DED experiments (exp04), three separate files that are combined with subfigures in LaTeX (each gets its own caption):
  fig01_dataset_snapshots          3D cuboid snapshots of the heat source (row 'Source') and the temperature (row 'Response') of one simulation (ID 12: 15 mm/s, 400 W) at four times (cf. Roth et al. 2025, Fig. 5)
  fig02_dataset_parameter_grid     the 5 x 5 grid (source speed v, source power Q) of the 25 simulations
  fig03_dataset_peak_temperature   peak temperature over time of all 25 simulations
Data: data/directed_energy_deposition/max_simulation (mesh 51 x 6 x 6 nodes, 50 x 10 x 10 mm, 25 simulations, 401 times over 20 s). The mesh is a tensor-product grid, pyvista is not needed.
Writes imgs/main/exp04/fig01..03 (.png, .pdf) (Figure 4.12).
Usage: python figures/make_dataset_figures.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parents[1] / "shared"))
import export_paths as EP
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import Normalize

import thesis_plot_style as tps

DATA = EP.REPO / "data/directed_energy_deposition/max_simulation"
CASE = 12                             # (15 mm/s, 400 W): centre of the 5 x 5 grid
TIMES = [0.5, 2.0, 4.0, 19.0]         # s
T_RANGE, Q_RANGE = (293.15, 1000.0), (0.0, 1.7e10)
CMAP_T, CMAP_Q = "coolwarm", "Oranges"

def grid_indices(points):
    """Sorted coordinate axes of the tensor-product mesh and the grid index of every node."""
    ax = [np.unique(points[:, i].round(9)) for i in range(3)]
    return ax, np.stack([np.searchsorted(ax[i], points[:, i].round(9)) for i in range(3)], axis=1)


def to_grid(values, idx, shape):
    """Node values on the 3D grid."""
    g = np.full(shape, np.nan); g[idx[:, 0], idx[:, 1], idx[:, 2]] = values
    return g


def draw(ax, axes_xyz, grid, cmap, norm):                              # (box aspect and view of the 3D body are set here; the zoom is set by the caller)
    """Colour the three visible faces (top z=0, front y=0, end x=max) with cell-averaged node values."""
    x, y, z = axes_xyz; mm = 1e3
    def face(X, Y, Z, F):
        c = 0.25 * (F[:-1, :-1] + F[1:, :-1] + F[:-1, 1:] + F[1:, 1:])
        ax.plot_surface(X * mm, Y * mm, Z * mm, facecolors=cmap(norm(c)), rstride=1, cstride=1, shade=False, linewidth=0.1, edgecolor=(0, 0, 0, 0.10), antialiased=False)
    X, Y = np.meshgrid(x, y, indexing="ij"); face(X, Y, np.full_like(X, z[-1]), grid[:, :, -1])
    X, Z = np.meshgrid(x, z, indexing="ij"); face(X, np.full_like(X, y[0]), Z, grid[:, 0, :])
    Y, Z = np.meshgrid(y, z, indexing="ij"); face(np.full_like(Y, x[-1]), Y, Z, grid[-1, :, :])
    ax.set_box_aspect((50, 10, 10), zoom=1.22); ax.view_init(elev=24, azim=-52); ax.set_axis_off()
    ax.set_xlim(0, 50); ax.set_ylim(0, 10); ax.set_zlim(-10, 0)


K1, K2, K3 = "fig01_dataset_snapshots", "fig02_dataset_parameter_grid", "fig03_dataset_peak_temperature"
GEN = "04_ded/figures/make_dataset_figures.py"
ZOOM = 1.62                           # the cuboid fills the width of its cell (box aspect 50 : 10 : 10 and the view are unchanged)


def fit_zoom(fig, ax, margin_px=2.0):
    """largest zoom of the 3D box (aspect 50 : 10 : 10, view unchanged) for which the projected cuboid lies inside the axes rectangle (bisection on the measured projected corners)."""
    from mpl_toolkits.mplot3d import proj3d
    corners = np.array([[x, y, z] for x in (0, 50) for y in (0, 10) for z in (-10, 0)], float)
    r = fig.canvas.get_renderer()

    def inside(z):
        ax.set_box_aspect((50, 10, 10), zoom=z); fig.canvas.draw()
        px, py, _ = proj3d.proj_transform(corners[:, 0], corners[:, 1], corners[:, 2], ax.get_proj())
        d = ax.transData.transform(np.column_stack([px, py])); bb = ax.get_window_extent(r)
        return d[:, 0].min() >= bb.x0 + margin_px and d[:, 0].max() <= bb.x1 - margin_px and d[:, 1].min() >= bb.y0 + margin_px and d[:, 1].max() <= bb.y1 - margin_px
    lo, hi = 0.5, 3.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if inside(mid) else (lo, mid)
    return lo


def snapshots(ds, axes_xyz, idx, shape, t):
    """Figure 4.12 (a): heat source (top row) and temperature (bottom row) of one simulation at four times."""
    tps.apply_style(K1)
    mm = 1 / 25.4
    W = tps.sizes.width_in(K1)
    ml, mr, cw, ch = 15.0 * mm, 0.5 * mm, None, 19.5 * mm
    cw = (W - ml - mr) / 4
    head, pitch, cb_h, cb_blk = 5.0 * mm, 14.5 * mm, 2.6 * mm, 11.0 * mm      # pitch < ch: the (transparent) axes of the rows overlap, the cuboids are 12 mm high
    H = head + ch + pitch + cb_blk + 1.0 * mm
    fig = plt.figure(figsize=(W, H))
    normT, normQ = Normalize(*T_RANGE), Normalize(*Q_RANGE); axs = {}
    for ri, (name, cmap, norm) in enumerate((("heat_source", plt.get_cmap(CMAP_Q), normQ), ("temperature", plt.get_cmap(CMAP_T), normT))):
        for k, tt in enumerate(TIMES):
            y0 = H - head - ch - ri * pitch
            ax = axs[ri, k] = fig.add_axes([(ml + k * cw) / W, y0 / H, cw / W, ch / H], projection="3d")
            draw(ax, axes_xyz, to_grid(ds[name].isel(batch=CASE, time=int(np.argmin(abs(t - tt)))).values, idx, shape), cmap, norm)
            ax.set_box_aspect((50, 10, 10), zoom=ZOOM)
            if ri == 0:
                tps.col_heading(ax, f"$t={tt:g}$ s", pad_effective=-6.0)
    zoom = fit_zoom(fig, axs[0, 0])                                   # largest zoom for which the whole cuboid stays inside its axes (nothing clipped)
    for a in axs.values():
        a.set_box_aspect((50, 10, 10), zoom=zoom)
    print(f"snapshot zoom {zoom:.3f}")
    y_cb = (H - head - ch - pitch + 1.25 * mm - cb_h) / H                       # bar 2.5 mm below the cuboids of the bottom row
    cq = fig.add_axes([(ml + 0.35 * cw) / W, y_cb, 1.6 * cw / W, cb_h / H]); c1 = fig.colorbar(cm.ScalarMappable(normQ, CMAP_Q), cax=cq, orientation="horizontal"); c1.set_label(r"$Q$ [$10^{10}$ W m$^{-3}$]")
    c1.set_ticks([0, 0.5e10, 1.0e10, 1.5e10]); c1.set_ticklabels(["0", "0.5", "1.0", "1.5"])
    ct = fig.add_axes([(ml + 2.35 * cw) / W, y_cb, 1.6 * cw / W, cb_h / H]); fig.colorbar(cm.ScalarMappable(normT, CMAP_T), cax=ct, orientation="horizontal", extend="max").set_label(r"$T$ [K]")
    tps.row_headings([([axs[0, k] for k in range(4)], "Source"), ([axs[1, k] for k in range(4)], "Response")], pad_effective=4.0, rotation=0)
    tps.save_figure(fig, EP.IMG_MAIN / K1, generator=GEN); plt.close(fig)


def main():
    """The three dataset figures: snapshots, parameter grid, peak temperature."""
    EP.ensure()
    ds = xr.open_dataset(DATA / "dataset.nc")
    axes_xyz, idx = grid_indices(ds.points.values); shape = tuple(len(a) for a in axes_xyz)
    t = ds.time.values; v = ds.source_speed.values; Q = ds.source_power.values
    snapshots(ds, axes_xyz, idx, shape, t)
    tps.apply_style(K2)                                                  # panel (b): square plot area, 0.38 of the text width
    fig, ax = plt.subplots(figsize=(tps.sizes.width_in(K2), tps.sizes.width_in(K2) * 0.92))
    ax.scatter(v, Q, s=16, color="0.25", zorder=2)
    ax.set_xlabel(r"$v$ [mm/s]"); ax.set_ylabel(r"$Q$ [W]"); ax.set_xticks([10, 12.5, 15, 17.5, 20]); ax.set_yticks([300, 350, 400, 450, 500]); ax.set_xlim(9, 21); ax.set_ylim(280, 520); ax.set_box_aspect(1)
    fig.tight_layout(); tps.save_figure(fig, EP.IMG_MAIN / K2, generator=GEN); plt.close(fig)
    Tmax = ds.temperature.max("node").values
    tps.apply_style(K3)                                                  # panel (c): wider and lower, 0.58 of the text width
    fig, ax = plt.subplots(figsize=(tps.sizes.width_in(K3), tps.sizes.width_in(K3) * 0.56))
    for b in range(25):
        ax.plot(t, Tmax[b], color="0.35", lw=0.6)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"Peak $T$ [K]"); ax.set_xlim(0, 20)
    fig.tight_layout(); tps.save_figure(fig, EP.IMG_MAIN / K3, generator=GEN); plt.close(fig)
    print("wrote", [f"{n}.{e}" for n in (K1, K2, K3) for e in ("pdf", "png")])


if __name__ == "__main__":
    main()
