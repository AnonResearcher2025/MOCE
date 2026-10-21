"""Compare six regression-model families for the three MOCE case studies.

Families: CART, Random Forest, Gradient Boosting, MLP, Linear Regression,
and Linear SVM (LinearSVR, since both outcomes are continuous).

Place next to Data-Skills.csv, Data-Big5.csv, Data-HRM.csv and run:
    python compare_predictive_models.py
    python compare_predictive_models.py --quick
    python compare_predictive_models.py --settings skills --repeats 3

For VS Code 'Run in Interactive Window', edit DEFAULT_SETTINGS / QUICK_TEST /
DEFAULT_REPEATS below: Jupyter kernel arguments are ignored automatically.

Each candidate is evaluated by *three repeated five-fold CVs*. We pool the
out-of-fold predictions within each repetition before computing R2, RMSE and
MAE separately for both outcomes. Candidates are ranked by the equally
weighted mean of the two outcome-specific average R2 scores.

CAUTION: The same CV selects the model/hyperparameters and reports its score.
The winner's CV score is therefore subject to selection optimism; use nested
CV or an untouched holdout for a less biased publication-level assessment.

This script only screens predictive models. Saving a model does NOT embed it
in the MOCE mathematical optimization; check the existing model-specific
constraint-extraction and embedding routines separately. Gradient Boosting
and LinearSVR are fitted *separately per outcome* through MultiOutputRegressor.
Preprocessing for scaled models is always learned inside the training fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

# =========================== USER SETTINGS ===========================
DEFAULT_SETTINGS = ("skills", "big_five", "hrm_practices")
DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 3
DEFAULT_SEED = 42
QUICK_TEST = False   # True: one five-fold CV and a small subset of candidates
SAVE_FITTED_MODELS = True
OUTPUT_DIR = "model_selection_six_models"  # separate from earlier results

DATASETS = {
    "skills": ("Data-Skills.csv", ("Performance", "Satisfaction")),
    "big_five": ("Data-Big5.csv", ("Performance", "Stress")),
    "hrm_practices": ("Data-HRM.csv", ("Performance", "Turnover Intention")),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    estimator: Any
    parameters: dict[str, Any]
    # This flag is deliberately strict: it means the *current* two-output
    # experiment representation can be passed to the existing case-study
    # pipeline without new preprocessing / model-wrapper conversions.
    directly_reusable: bool


def build_candidates(seed: int, quick: bool) -> list[Candidate]:
    specs: list[Candidate] = []

    def add(name: str, family: str, est: Any, params: dict, direct: bool = False):
        specs.append(Candidate(name, family, est, params, direct))

    # 1. CART: include the original study configuration (depth 6, leaf 3).
    tree_grid = ([(6, 3)] if quick else
                 [(3, 3), (5, 3), (6, 3), (8, 3),
                  (3, 6), (5, 6), (8, 6), (None, 10)])
    for depth, leaf in tree_grid:
        params = {"max_depth": depth, "min_samples_leaf": leaf}
        add(f"CART_d{depth}_leaf{leaf}", "CART",
            DecisionTreeRegressor(random_state=seed, **params), params, True)

    # 2. RF: native multi-output sklearn regressor.
    rf_grid = ([(4, 6)] if quick else
               [(4, 3), (4, 6), (6, 3), (6, 6)])
    for depth, leaf in rf_grid:
        params = {"n_estimators": 150, "max_depth": depth,
                  "min_samples_leaf": leaf, "max_features": 1.0}
        add(f"RF_d{depth}_leaf{leaf}", "RF",
            RandomForestRegressor(random_state=seed, n_jobs=1, **params), params, True,)

    # 3. GBM: sklearn's GradientBoostingRegressor supports one outcome;
    #    MultiOutputRegressor trains one independent boosted ensemble per target.
    gbm_grid = ([(2, 100, 0.1)] if quick else
                [(2, 100, 0.05), (2, 200, 0.05), (2, 100, 0.1),
                 (3, 100, 0.05), (3, 200, 0.05), (3, 100, 0.1)])
    for depth, ntrees, rate in gbm_grid:
        params = {"max_depth": depth, "n_estimators": ntrees,
                  "learning_rate": rate, "min_samples_leaf": 3}
        add(f"GBM_d{depth}_n{ntrees}_lr{rate:g}", "GBM",
            MultiOutputRegressor(GradientBoostingRegressor(
                random_state=seed, **params), n_jobs=1), params)

    # 4. MLP: raw and scaled are both from the single MLP family.
    #    Raw ReLU-MLP resembles the current MOCE neural-network input path.
    raw_mlp_grid = (
        [(20, 0.0001)] if quick else
        [(width, alpha)
        for width in [10, 20]
        for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]]
    )
    for width, alpha in raw_mlp_grid:
        params = {"hidden_layer_sizes": [width], "alpha": alpha,
                  "activation": "relu", "solver": "lbfgs", "scaling": "none"}
        add(f"MLP_raw_h{width}_a{alpha:g}", "MLP",
            MLPRegressor(hidden_layer_sizes=(width,), alpha=alpha,
                         activation="relu", solver="lbfgs", max_iter=1200,
                         random_state=seed), params, True)

    # 5. Ordinary linear regression: fit one coefficient vector per outcome.
    add("LinearRegression_OLS", "LinearRegression", LinearRegression(), {}, False)

    # 6. Linear SVM regression: LinearSVR + one model per target.
    #    StandardScaler is fitted within each CV training fold.
    svm_grid = ([(1.0, 0.1)] if quick else
                [(0.1, 0.1), (1.0, 0.1), (10.0, 0.1),
                 (0.1, 0.5), (1.0, 0.5), (10.0, 0.5)])
    for C, eps in svm_grid:
        params = {"C": C, "epsilon": eps, "loss": "squared_epsilon_insensitive",
                  "scaling": "StandardScaler (within CV fold)"}
        add(f"LinearSVM_C{C:g}_eps{eps:g}", "LinearSVM",
            make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(LinearSVR(
                    C=C, epsilon=eps, loss="squared_epsilon_insensitive",
                    dual=False, max_iter=20000, random_state=seed), n_jobs=1)),
            params)

    return specs


def find_project_root() -> Path:
    if "__file__" in globals():
        location = Path(__file__).resolve().parent
        if (location / "Data-Skills.csv").exists():
            return location
    return Path.cwd().resolve()


def load_data(root: Path, slug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    filename, outcome_cols = DATASETS[slug]
    location = root / filename
    if not location.is_file():
        raise FileNotFoundError(f"Missing {location}. Put script in the data folder.")
    df = pd.read_csv(location)
    if df.columns[-2:].tolist() != list(outcome_cols):
        raise ValueError(f"Unexpected targets in {filename}: "
                         f"expected last two columns {outcome_cols}, "
                         f"found {df.columns[-2:].tolist()}")
    if len(df) < DEFAULT_FOLDS:
        raise ValueError(f"Insufficient rows in {filename}: {len(df)}")
    if df.isna().any().any():
        raise ValueError(f"Missing values found in {filename}")
    if any(not pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        raise ValueError(f"Nonnumeric predictors/targets in {filename}")
    return df.iloc[:, :-2].copy(), df.loc[:, list(outcome_cols)].copy()


def evaluate(spec: Candidate, X: pd.DataFrame, y: pd.DataFrame,
             splits: list, folds: int, repeats: int) -> list[dict]:
    true = y.to_numpy(dtype=float)
    rows = []
    for rep in range(repeats):
        pred = np.full_like(true, np.nan)
        for tr, te in splits[rep * folds:(rep + 1) * folds]:
            model = clone(spec.estimator)
            model.fit(X.iloc[tr], y.iloc[tr])
            fold_pred = np.asarray(model.predict(X.iloc[te]), dtype=float)
            if fold_pred.shape != (len(te), true.shape[1]):
                raise ValueError(f"Unexpected prediction shape: {fold_pred.shape}")
            pred[te] = fold_pred
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Invalid out-of-fold predictions: {spec.name}")
        r2s = r2_score(true, pred, multioutput="raw_values")
        maes = mean_absolute_error(true, pred, multioutput="raw_values")
        rmses = np.sqrt(mean_squared_error(true, pred, multioutput="raw_values"))
        for j, outcome in enumerate(y.columns):
            rows.append({"candidate": spec.name, "family": spec.family,
                         "repeat": rep + 1, "outcome": outcome,
                         "r2": float(r2s[j]), "rmse": float(rmses[j]),
                         "mae": float(maes[j])})
    return rows


def summarize(detail: pd.DataFrame, successful: dict[str, Candidate],
              targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_outcome = (detail.groupby(["candidate", "family", "outcome"], as_index=False)
                   .agg(mean_r2=("r2", "mean"), sd_r2=("r2", "std"),
                        mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"),
                        mean_mae=("mae", "mean")))
    for col in ["sd_r2", "sd_rmse"]:
        per_outcome[col] = per_outcome[col].fillna(0.0)
    wide = per_outcome.pivot(index="candidate", columns="outcome", values="mean_r2")
    rows = []
    for name, spec in successful.items():
        vals = [float(wide.loc[name, col]) for col in targets]
        rows.append({"candidate": name, "family": spec.family,
                     "directly_reusable_in_current_moce": spec.directly_reusable,
                     f"r2_{targets[0]}": vals[0],
                     f"r2_{targets[1]}": vals[1],
                     "r2_outcome_1": vals[0], "r2_outcome_2": vals[1],
                     "mean_r2": float(np.mean(vals)),
                     "min_r2": float(np.min(vals)),
                     "both_r2_positive": bool(all(v > 0 for v in vals)),
                     "parameters": json.dumps(spec.parameters, sort_keys=True)})
    ranked = pd.DataFrame(rows).sort_values(
        ["mean_r2", "min_r2"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked, per_outcome


def run_setting(root: Path, slug: str, args: argparse.Namespace) -> list[dict]:
    X, y = load_data(root, slug)
    folder = root / "results" / OUTPUT_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)
    specs = build_candidates(args.seed, args.quick)
    baseline = Candidate("Mean_baseline", "MeanBaseline",
                         DummyRegressor(strategy="mean"), {}, False)
    all_specs = [baseline, *specs]
    by_name = {c.name: c for c in all_specs}
    cv = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats,
                       random_state=args.seed)
    splits = list(cv.split(X))
    print(f"\n{'='*70}\n{slug}: N={len(X)}, predictors={X.shape[1]}, "
          f"outcomes={list(y.columns)}", flush=True)
    print(f"{len(specs)} candidate configurations across SIX families; "
          f"{args.repeats} x {args.folds}-fold CV", flush=True)
    detail_rows = []
    errors = []
    for k, spec in enumerate(all_specs, 1):
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                rows = evaluate(spec, X, y, splits, args.folds, args.repeats)
            detail_rows.extend(rows)
            cv_warnings = sum(issubclass(w.category, ConvergenceWarning)
                              for w in recorded)
            r2 = (pd.DataFrame(rows).groupby("outcome", sort=False)["r2"]
                  .mean().reindex(y.columns).tolist())
            print(f" [{k:02}/{len(all_specs):02}] {spec.name:<29} "
                  f"R2=({r2[0]:+.3f}, {r2[1]:+.3f}) "
                  f"mean={np.mean(r2):+.3f}; warnings={cv_warnings} "
                  f"[{time.perf_counter()-start:.1f}s]", flush=True)
        except Exception as exc:
            errors.append({"candidate": spec.name, "family": spec.family,
                           "error": repr(exc)})
            print(f" [{k:02}/{len(all_specs):02}] {spec.name}: FAILED: {exc}",
                  flush=True)

    if not detail_rows:
        raise RuntimeError(f"Every candidate failed for {slug}")
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(folder / "repeat_scores.csv", index=False)
    successful = {c.name: c for c in all_specs if c.name in set(detail.candidate)}
    ranked, per_outcome = summarize(detail, successful, list(y.columns))
    ranked.to_csv(folder / "all_candidates.csv", index=False)
    per_outcome.to_csv(folder / "per_outcome_scores.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(folder / "candidate_errors.csv", index=False)
    elif (folder / "candidate_errors.csv").exists():
        (folder / "candidate_errors.csv").unlink()

    valid = ranked[ranked.family != "MeanBaseline"].copy()
    if valid.empty:
        raise RuntimeError(f"No candidate predictive model fitted for {slug}")
    chosen: list[tuple[str, pd.Series]] = [("best_overall", valid.iloc[0])]
    both_positive = valid[valid.both_r2_positive]
    if not both_positive.empty:
        chosen.append(("best_both_positive", both_positive.iloc[0]))
    direct = valid[valid.directly_reusable_in_current_moce]
    if not direct.empty:
        chosen.append(("best_direct_moce", direct.iloc[0]))
    for family in ["CART", "RF", "GBM", "MLP", "LinearRegression", "LinearSVM"]:
        subset = valid[valid.family == family]
        if not subset.empty:
            chosen.append((f"best_{family.lower()}", subset.iloc[0]))
    family_best = pd.concat([valid[valid.family == family].head(1)
                             for family in ["CART", "RF", "GBM", "MLP",
                                            "LinearRegression", "LinearSVM"]
                             if (valid.family == family).any()])
    family_best.to_csv(folder / "best_by_family.csv", index=False)

    selected = []
    for label, row in chosen:
        spec = by_name[str(row.candidate)]
        rec = {"setting": slug, "selection": label, "candidate": spec.name,
               "family": spec.family, "mean_r2": float(row.mean_r2),
               "min_r2": float(row.min_r2),
               "both_r2_positive": bool(row.both_r2_positive),
               "r2_outcome_1": float(row.r2_outcome_1),
               "r2_outcome_2": float(row.r2_outcome_2),
               "directly_reusable_in_current_moce": spec.directly_reusable,
               "parameters": json.dumps(spec.parameters, sort_keys=True)}
        selected.append(rec)
        if SAVE_FITTED_MODELS:
            trained = clone(spec.estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                trained.fit(X, y)
            joblib.dump(trained, folder / f"{label}.joblib")
            metadata = {**rec, "input_columns_ordered": list(X.columns),
                        "outcome_names": list(y.columns),
                        "number_training_observations": int(len(X)),
                        "fit_note": "Refitted on entire original dataset after CV selection",
                        "optimization_note": "Additional checking/integration is required before using a newly selected model in the MOCE MILP."}
            (folder / f"{label}_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(selected).to_csv(folder / "selected_models.csv", index=False)
    print(f" >> Best overall: {valid.iloc[0].candidate} "
          f"(mean R2={valid.iloc[0].mean_r2:+.3f})")
    print(" >> Best with positive R2 on both outcomes: " +
          (str(both_positive.iloc[0].candidate) if not both_positive.empty else "none"))
    print(f" >> Saved output: {folder}", flush=True)
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOCE predictive comparison: six regressors")
    p.add_argument("--settings", nargs="+", choices=list(DATASETS),
                   default=list(DEFAULT_SETTINGS))
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--quick", action="store_true", default=QUICK_TEST)
    args = p.parse_args([] if "ipykernel" in sys.modules else None)
    if args.folds < 2 or args.repeats < 1:
        p.error("--folds must be >=2, --repeats must be >=1")
    if args.quick:
        args.repeats = 1
    return args


def main() -> None:
    args = parse_args()
    root = find_project_root()
    print("Data directory:", root)
    print("Ranking: mean of both outcomes' mean repeated out-of-fold R2")
    print("CAUTION: selection CV scores are not independent validation scores")
    selections: list[dict] = []
    for setting in args.settings:
        selections.extend(run_setting(root, setting, args))
    out = root / "results" / OUTPUT_DIR
    pd.DataFrame(selections).to_csv(out / "all_settings_selected_models.csv",
                                     index=False)
    print("\nFinished; original case-study code and results were not changed.")


if __name__ == "__main__":
    main()
