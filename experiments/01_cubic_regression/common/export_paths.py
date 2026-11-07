"""Where the cubic-regression thesis exports are written (single definition).

FINAL = thesis_figures/final/   mirrors the paths of the thesis project (folder 00_Masterthesis/)
AUDIT = thesis_figures/final_audit/   numbers behind Figure 4.2 (ffnn_summary.json); not a thesis input
Environment overrides (smoke tests): CR_EXPORT_ROOT, CR_AUDIT_ROOT.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS_FIGURES = ROOT / "thesis_figures"
FINAL = Path(os.environ.get("CR_EXPORT_ROOT", THESIS_FIGURES / "final"))
AUDIT = Path(os.environ.get("CR_AUDIT_ROOT", THESIS_FIGURES / "final_audit"))
KB = FINAL / "00_Masterthesis" / "Knowledgebase"
IMG_MAIN, IMG_APP = KB / "imgs" / "main" / "exp01", KB / "imgs" / "appendix" / "exp01"
TAB_MAIN, TAB_APP = KB / "tables" / "main" / "exp01", KB / "tables" / "appendix" / "exp01"
ALL_DIRS = (IMG_MAIN, IMG_APP, TAB_MAIN, TAB_APP, AUDIT)


def ensure():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
