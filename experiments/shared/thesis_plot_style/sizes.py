"""Included-width manifest and font-size calibration.

    scale               = included_width / exported_PDF_box_width
    effective_font_size = exported_font_size * scale
    exported_font_size  = target_effective_font_size / scale

Policy of the generators: export AT the inclusion width. Fonts are the target sizes (exported size = target size), the FIGURE width is adjusted from run to run so that the exported PDF box
(which `bbox_inches="tight"` makes slightly different from the figure width) equals the inclusion width; scale is then 1 within 1 % and the effective sizes equal the targets.
`width_in(key)` returns the figure width to use (= inclusion width, corrected with the box of the previous export). The formulas above stay valid for a PDF that is included at another width
than it was exported at: effective = exported * scale.

figure_widths.json    INPUT, edited by hand once the LaTeX inclusion widths are fixed. Key = file stem of the exported PDF;
                      "included_mm" = resolved physical width of the figure in the thesis (a nested subfigure counts with its own width, not with the literal multiplier).
size_calibration.json OUTPUT of `save_figure`: the measured PDF box (MediaBox as written by Matplotlib = CropBox of the file), the resulting scale and the raw / effective
                      font size of every text role. `apply_style(key)` reads the box of the previous export of the same figure, so that a second run converges
                      (the tight bounding box changes slightly with the font size).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTHS_FILE = HERE / "figure_widths.json"
CALIB_FILE = HERE / "size_calibration.json"
MM_PER_BP = 25.4 / 72.0                      # Matplotlib / PDF points (bp); TeX points differ by 0.4 percent, negligible here
TEXTWIDTH_MM = 160.0                         # A4 minus two 25 mm margins


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def widths() -> dict:
    return _load(WIDTHS_FILE)


def included_mm(key: str) -> float:
    return float(widths()[key]["included_mm"])


def width_in(key: str) -> float:
    """Figure width in inches to draw the figure with: the figure width of the previous export of this figure corrected by (included - exported PDF box)
    (the tight bounding box crops or extends the figure), so that the exported box converges to the inclusion width."""
    w = included_mm(key) / 25.4
    c = _load(CALIB_FILE).get(key)
    if c and c.get("box_mm") and c.get("figsize_mm"):
        w = (c["figsize_mm"][0] + (included_mm(key) - float(c["box_mm"]))) / 25.4        # additive correction: the tight box differs from the figure width by (mostly absolute) label / heading extents
    return w


def scale(key: str | None) -> float:
    """included width / PDF box of the previous export of this figure (1.0 if the figure is not in the manifest or was not exported yet); report only, see the policy above."""
    if not key:
        return 1.0
    w, c = widths().get(key), _load(CALIB_FILE).get(key)
    if not w or not c or not c.get("box_mm"):
        return 1.0
    return float(w["included_mm"]) / float(c["box_mm"])


def pdf_box_mm(pdf: Path) -> tuple[float, float]:
    """(width, height) in mm of the PDF page box (Matplotlib writes MediaBox only; it is the box LaTeX uses)."""
    raw = Path(pdf).read_bytes()[:20000]
    m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", raw)
    if m is None:
        raise ValueError(f"no MediaBox in {pdf}")
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return (x1 - x0) * MM_PER_BP, (y1 - y0) * MM_PER_BP


def record(key: str, pdf: Path, figsize_in, raw_pt: dict, generator: str = "") -> dict | None:
    """Writes the measured box and the effective sizes of `key` into size_calibration.json; returns the entry (None if the figure is not in the manifest)."""
    w = widths().get(key)
    if w is None:
        return None
    wmm, hmm = pdf_box_mm(pdf)
    s = float(w["included_mm"]) / wmm
    entry = {"path": w.get("path", ""), "included_mm": float(w["included_mm"]), "status": w.get("status", "provisional"), "box_mm": round(wmm, 2), "box_height_mm": round(hmm, 2),
             "figsize_mm": [round(figsize_in[0] * 25.4, 2), round(figsize_in[1] * 25.4, 2)], "scale": round(s, 4),
             "raw_pt": {k: round(v, 2) for k, v in raw_pt.items()}, "effective_pt": {k: round(v * s, 2) for k, v in raw_pt.items()}, "generator": generator}
    if os.environ.get("TPS_CALIBRATION_WRITE"):                      # the calibration state is only updated when explicitly requested
        calib = _load(CALIB_FILE)
        calib[key] = entry
        CALIB_FILE.write_text(json.dumps(dict(sorted(calib.items())), indent=1) + "\n")
    return entry
