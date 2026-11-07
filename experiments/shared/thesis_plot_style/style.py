"""Single source of truth for colours, line styles, markers, labels and sizes of all thesis experiment figures (harmonic oscillator, cubic regression, ...).
Rules for all thesis figures. Nothing in a plotting script may hard-code a colour, line style or marker of the roles below.

1. A figure never relies on colour alone: every method has its own marker, every line role its own line style.
2. Colour families (never mixed within one panel):
   - METHODS (comparison plots)            Paul Tol "muted" palette, fixed per method
   - STATE VARIABLES (function space)      q blue, p orange (as in Ott et al.)
   - ANALYSIS CATEGORIES (terms, measures) Okabe-Ito
   - UNCERTAINTY MAPS                      sequential "Blues", per-panel scale unless stated
3. Line-style roles: solid = model estimate; dashed black = ground truth or theory reference; dotted = a level / single members;
   dash-dot grey = characteristic time.
Colour-blind safety was checked with the Machado et al. (2009) simulations for deuteranopia, protanopia and tritanopia.
"""

TEXTWIDTH_IN = 6.3          # assumed text width of the thesis; change here only

# ---- typography at the FINAL document size: effective sizes after inclusion in LaTeX; normal weight everywhere ---------------------
# raw (exported) size = target / scale, scale = included width / exported PDF box width (sizes.py); the same roles are used for every figure.
TARGET_PT = {"label": 9,      # axis labels and colour-bar labels
             "heading": 9,    # row and column headings (outside the data rectangles)
             "tick": 8,       # tick labels
             "legend": 8,     # legends
             "annot": 8,      # numerical cell annotations
             "flow": 9}       # flowchart text
ROW_HEADING_ROTATION = 90   # row headings are rotated like the y axis labels; column headings are horizontal
HEADING_PAD_PT = 3.0        # effective distance between a heading and its panel / between a row heading and the left-most tick labels or axis label
LEGEND_GAP_PT = 3.0         # effective gap between the shared legend and the highest heading / axes content

# ---- methods ---------------------------------------------------------------------------------------------------------
METHODS = ("LA", "MoLA", "DE", "SWAG", "MultiSWAG")
METHOD_COLOR = {"LA": "#4477AA", "MoLA": "#CC6677", "DE": "#228833", "SWAG": "#AA3377", "MultiSWAG": "#66CCEE"}
METHOD_MARKER = {"LA": "o", "MoLA": "s", "DE": "^", "SWAG": "D", "MultiSWAG": "v"}
METHOD_LABEL = {"LA": "LA", "MoLA": "MoLA", "DE": "DE", "SWAG": "SWAG", "MultiSWAG": "MultiSWAG"}

# ---- state variables (function-space panels) -------------------------------------------------------------------------
Q_COLOR, P_COLOR = "#4c78a8", "#f58518"

# ---- analysis categories (Okabe-Ito) ----------------------------------------------------------------------------------
ROLE_A, ROLE_B, ROLE_C, ROLE_D = "#0072B2", "#D55E00", "#009E73", "#CC79A7"   # blue, vermillion, bluish green, reddish purple
BLACK = "0.12"

# ---- line roles ---------------------------------------------------------------------------------------------------------
TRUTH = dict(color=BLACK, ls="--", lw=0.8)                      # true trajectory / true orbit; also theoretical predictions
LEVEL = dict(color="k", ls=":", lw=0.9)                         # saturation level a/sqrt(2), other reference levels
CHAR_TIME = dict(color="0.45", ls="-.", lw=1.0)                 # characteristic times, e.g. t* = 1/sigma_Omega
MEAN_COLOR = "#E69F00"                                          # predictive mean in the phase plane (Okabe-Ito orange; red/black is not safe)
MEMBER = dict(color="#762a83", ls=(0, (1, 1.5)), lw=0.5)        # single members / mixture components (supplementary figures only)
ISO_COLOR = "#D55E00"                                           # isolines of the vector-field uncertainty (vector-field maps only, not in trajectory plots)
DATA_COLOR = "0.3"                                              # training observations (grey dots)

CMAP_UNCERTAINTY = "Blues"

# ---- line widths / marker sizes (de-facto values of the harmonic-oscillator figures, now the standard) --------------------------------
LW = {"main": 1.3, "mean": 1.1, "truth": 0.8, "reference": 0.9, "thin": 0.5, "grid": 0.4}
MARKER_SIZE = 3.0

# ---- figure widths as multiples of TEXTWIDTH_IN ---------------------------------------------------------------------------------------
WIDTH_FACTORS = {"text": 1.0, "wide": 1.35, "slightly_wide": 1.15}   # 'wide' = landscape / full-page figures
