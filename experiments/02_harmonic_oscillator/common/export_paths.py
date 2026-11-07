"""Where the harmonic-oscillator thesis exports are written (single definition; all export scripts import this).

FINAL  = thesis_figures/final/   mirrors the paths of the thesis project (folder 00_Masterthesis/)
AUDIT  = thesis_figures/final_audit/   csv / md / provenance / reproduction checks that are not thesis inputs
Environment overrides (smoke tests, writing to a scratch directory instead of the final folder): HO_EXPORT_ROOT, HO_AUDIT_ROOT.
Usage in a script:  sys.path.insert(0, str(ROOT / "common")); import export_paths as EP
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS_FIGURES = ROOT / "thesis_figures"
FINAL = Path(os.environ.get("HO_EXPORT_ROOT", THESIS_FIGURES / "final"))
AUDIT = Path(os.environ.get("HO_AUDIT_ROOT", THESIS_FIGURES / "final_audit"))
KB = FINAL / "00_Masterthesis" / "Knowledgebase"
IMG_MAIN, IMG_APP = KB / "imgs" / "main" / "exp02", KB / "imgs" / "appendix"
TAB_MAIN, TAB_APP = KB / "tables" / "main" / "exp02", KB / "tables" / "appendix"
ALL_DIRS = (IMG_MAIN, IMG_APP, TAB_MAIN, TAB_APP, AUDIT)


def ensure():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
