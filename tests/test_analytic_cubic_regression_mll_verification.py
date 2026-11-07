from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_verification_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "01_cubic_regression"
        / "verify_analytic_cubic_regression_mll.py"
    )
    spec = importlib.util.spec_from_file_location("analytic_cubic_regression_mll", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analytic_cubic_regression_mll_verification_core(tmp_path):
    module = _load_verification_module()
    summary = module.run_verification(
        output_dir=tmp_path,
        make_plots=False,
        write_summary=False,
    )

    fixed = summary["fixed_hyperparameter_laplace"]
    assert fixed["status"] == "PASS"
    assert fixed["ggn_relative_error"] < 1e-12
    assert fixed["posterior_covariance_relative_error"] < 1e-12
    assert fixed["predictive_variance_relative_error"] < 1e-12
    assert fixed["predictive_dense_covariance_relative_error"] < 1e-12

    mll = summary["mll_objective_fixed_pairs"]
    assert mll["status"] == "PASS"
    assert mll["max_absolute_difference"] < 1e-10

    adam = summary["repo_adam_calibration"]
    assert adam["best_adam_to_repo_fixed_objective_log_mll_gap"] < 1e-6
    assert adam["status_vs_exact_blr_empirical_bayes"] == (
        "FAIL_EXPECTED_FIXED_MAP_OBJECTIVE_DIFFERS"
    )
