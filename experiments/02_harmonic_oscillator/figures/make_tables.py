"""Harmonic oscillator: SWAG learning-rate table (Table B.7) and the authored settings tables (Tables B.3-B.6).

Format of every .tex: LaTeX `tabular` only (no table float, caption, label, resizebox), booktabs rules, plain text + math mode + \\textbf / \\mathbf only, so that the thesis can wrap it with
\\resizebox{\\linewidth}{!}{\\input{...}}. Number format (`num`): 3 significant digits, scientific as a{\\cdot}10^{b} for |x| < 1e-2 and |x| >= 1e3.

Outputs (paths from common/export_paths.py):
  ho_swag_dependence.tex     generated from the stored SWAG sweep (csv / md go to the audit folder)
  ho_structural_{setup,models,uq_settings}.tex, ott_qualitative_settings.tex     authored tables of tables/authored/, copied unchanged
Usage: python -u make_tables.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "swag_dependence"))
sys.path.insert(0, str(ROOT / "common"))
import export_paths as EP
import aggregate as AG  # (sweep_table / de_reference / cfg of the SWAG study; module has a __main__ guard)

import numpy as np

L = AG.L
AUTHORED_DIR = HERE / "tables" / "authored"
LRS = L.LR_GRID
DATASETS, MODELS = ("lower_half", "left_half"), ("NODE", "sPHNN")
AUTHORED = ("ho_structural_setup", "ho_structural_models", "ho_structural_uq_settings", "ott_qualitative_settings")


# ---------------------------------------------------------------- formatting
def sci_parts(x):
    """Mantissa and decimal exponent of x in scientific notation with two decimals."""
    m, e = f"{x:.2e}".split("e")
    return float(m), int(e)


def num(x, bold=False):
    """3 significant digits; scientific a{\\cdot}10^{b} below 1e-2 and from 1e3 on. Returns math-mode string."""
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        s = "0"
    else:
        m, e = sci_parts(abs(x))
        sign = "-" if x < 0 else ""
        if e < -2 or e >= 3:
            s = f"{sign}{m:.2f}{{\\cdot}}10^{{{e}}}"
        else:
            dec = max(0, 2 - e)
            s = f"{sign}{abs(x):.{dec}f}"
            if len(s.replace('-', '').replace('.', '').lstrip('0')) > 3 and dec > 0:   # rounding carried over (9.996 -> 10.00)
                s = f"{sign}{abs(x):.{max(0, dec - 1)}f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def write(name, dests, tex_lines, header, rows, md_lines=None):
    """Write <name>.tex into every destination; csv and md of the thesis folder go to the audit folder."""
    for d in dests:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.tex").write_text("\n".join(tex_lines) + "\n")
        if d in (EP.TAB_MAIN, EP.TAB_APP):          # the thesis folder receives the .tex only; csv / md are audit material
            d = EP.AUDIT
            d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        (d / f"{name}.md").write_text("\n".join(md_lines or md) + "\n")


# ---------------------------------------------------------------- SWAG lr table
def swag_data():
    """Per (model, data set): the sweep results per continuation rate and the DE reference."""
    data = {}
    for ds in DATASETS:
        for m in MODELS:
            members = L.sweep_members(m, ds)
            per = {}
            for lr in LRS:
                cs = [AG.cfg(m, ds, k, lr) for k in members]
                ok = [c for c in cs if c is not None and c.get("n_kept", 0) > 0 and np.isfinite(c.get("H_q", np.nan))]
                per[lr] = {"n_valid": len(ok), "n": len(cs), "std": [0.5 * (c["H_q"] + c["H_p"]) for c in ok]}
            de = AG.de_reference(m, ds)
            data[(m, ds)] = {"members": members, "per": per, "de": 0.5 * (de["H_q"] + de["H_p"])}
    return data


def lr_header(lr):
    """column heading of a continuation rate in the compact manuscript table: $10^{-5}$ ... $10^0$, $10^1$; the rate of the comparisons in bold"""
    e = int(round(np.log10(lr)))
    p = f"10^{{{e}}}" if e < 0 else f"10^{e}"
    return f"$\\mathbf{{{p}}}$" if np.isclose(lr, L.ens_lib.SWAG_LR) else f"${p}$"


def swag_tables(data):
    """ho_swag_dependence.tex: rows NODE / sPHNN on both data sets, columns the seven continuation rates 10^-5 ... 10^1 and the DE reference. Entries are the median over the sweep
    members (0-3) of <Std> = 0.5(<Std[q]> + <Std[p]>); the DE reference uses all retained members. The rate of the comparisons (10^-1) is bold; `Failed` = SWAG fit / rollouts failed for all members."""
    name_ac = {"NODE": "\\Ac{NODE}", "sPHNN": "\\Ac{sPHNN}"}
    tex = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}ll" + "r" * (len(LRS) + 1) + "@{}}", "\\toprule"]
    hdr = [lr_header(lr) for lr in LRS]
    tex += ["Model & Dataset & " + " & ".join(hdr[:4]), "  & " + " & ".join(hdr[4:]) + " & \\ac{DE} \\\\", "\\midrule"]
    header = ["model", "dataset"] + [f"lr={lr:g}" for lr in LRS] + ["DE"]
    rows = []
    for mi, m in enumerate(MODELS):
        for ds in DATASETS:
            d = data[(m, ds)]
            cells, csvc = [], []
            for lr in LRS:
                p = d["per"][lr]
                if p["n_valid"] == 0:
                    cells.append("Failed"); csvc.append("Failed")
                    continue
                v = float(np.median(p["std"]))
                cell = num(v, bold=bool(np.isclose(lr, L.ens_lib.SWAG_LR)))
                if p["n_valid"] < p["n"]:
                    raise ValueError(f"partially failed sweep cell {m}/{ds}/{lr}: {p['n_valid']}/{p['n']} members; the compact table has no footnote for it")
                cells.append(cell); csvc.append(f"{v:.6g}")
            cells.append(num(d["de"])); csvc.append(f"{d['de']:.6g}")
            k = 4 if not any("cdot" in c for c in cells[:4]) else 3                 # line break as in the manuscript fragment (four short values, or three when scientific notation is involved)
            tex += [f"{name_ac[m]} & \\emph{{{ds.replace('_', ' ')}}}", "  & " + " & ".join(cells[:k]), "  & " + " & ".join(cells[k:]) + " \\\\"]
            rows.append([m, ds] + csvc)
        if mi == 0:
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular*}"]
    write("ho_swag_dependence", [EP.TAB_APP], tex, header, rows)


# ---------------------------------------------------------------- authored tables
def export_authored():
    """Copy the authored tables (shared data / training / solver settings, NODE / sPHNN architectures, inference settings, Ott check settings) to the thesis export unchanged."""
    EP.TAB_APP.mkdir(parents=True, exist_ok=True)
    for n in AUTHORED:
        shutil.copyfile(AUTHORED_DIR / f"{n}.tex", EP.TAB_APP / f"{n}.tex")


if __name__ == "__main__":
    swag_tables(swag_data())
    export_authored()
