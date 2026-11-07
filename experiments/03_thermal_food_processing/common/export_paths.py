"""Output locations of the figure and table generators of the thermal food processing experiment and the location of the stored run records.

FINAL = thesis_figures/final/         mirrors the paths of the thesis project (folder 00_Masterthesis/); exp03 = thermal food processing
AUDIT = thesis_figures/final_audit/   numbers behind the figures (figure_data/), not thesis inputs
RESULTS_ROOT = runs/03_AL_chicken/    stored run records (read-only)
Environment overrides: CHICKEN_EXPORT_ROOT, CHICKEN_AUDIT_ROOT, CHICKEN_RESULTS_ROOT.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                       # experiments/03_thermal_food_processing
REPO = ROOT.parents[1]                                              # repository root
RESULTS_ROOT = Path(os.environ.get("CHICKEN_RESULTS_ROOT", REPO / "runs" / "03_AL_chicken"))
THESIS_FIGURES = ROOT / "thesis_figures"
FINAL = Path(os.environ.get("CHICKEN_EXPORT_ROOT", THESIS_FIGURES / "final"))
AUDIT = Path(os.environ.get("CHICKEN_AUDIT_ROOT", THESIS_FIGURES / "final_audit"))
KB = FINAL / "00_Masterthesis" / "Knowledgebase"
IMG_MAIN, IMG_APP = KB / "imgs" / "main" / "exp03", KB / "imgs" / "appendix" / "exp03"
TAB_MAIN, TAB_APP = KB / "tables" / "main" / "exp03", KB / "tables" / "appendix" / "exp03"
ALL_DIRS = (IMG_MAIN, IMG_APP, TAB_MAIN, TAB_APP, AUDIT)


def ensure():
    """Create the output folders."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
