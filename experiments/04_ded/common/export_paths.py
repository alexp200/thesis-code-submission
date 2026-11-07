"""Output locations of the figure and table generators of the DED experiment.

FINAL = thesis_figures/final/         mirrors the paths of the thesis project (folder 00_Masterthesis/); exp04 = DED active learning
AUDIT = thesis_figures/final_audit/   numbers behind the figures (figure_data/), not thesis inputs
Environment overrides: DED_EXPORT_ROOT, DED_AUDIT_ROOT.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                       # experiments/04_ded
REPO = ROOT.parents[1]                                              # repository root
THESIS_FIGURES = ROOT / "thesis_figures"
FINAL = Path(os.environ.get("DED_EXPORT_ROOT", THESIS_FIGURES / "final"))
AUDIT = Path(os.environ.get("DED_AUDIT_ROOT", THESIS_FIGURES / "final_audit"))
KB = FINAL / "00_Masterthesis" / "Knowledgebase"
IMG_MAIN, IMG_APP = KB / "imgs" / "main" / "exp04", KB / "imgs" / "appendix" / "exp04"
TAB_MAIN, TAB_APP = KB / "tables" / "main" / "exp04", KB / "tables" / "appendix" / "exp04"
ALL_DIRS = (IMG_MAIN, IMG_APP, TAB_MAIN, TAB_APP, AUDIT)


def ensure():
    """Create the output folders."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
