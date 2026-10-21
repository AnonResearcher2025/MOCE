"""Shared evaluation utilities for the three empirical MOCE scripts.

The functions in this file add a compact artifact-evaluation layer without
changing the original counterfactual optimization formulation. They cover:
- predictive adequacy against an outcome-mean baseline;
- single-outcome CE, sequential CE, and joint MOCE comparisons;
- threshold sensitivity; and
- feasibility, change burden, sparsity, and computation-time summaries.
"""

from __future__ import annotations

import json
import time
import hashlib
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def ensure_output_dir(path: Union[str, Path]) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _checkpoint_data_signature(X: pd.DataFrame, baseline_predictions: pd.DataFrame, outcome_names: Sequence[str]) -> str:
    """Identify the dataset and fitted predictions, avoiding stale-model resumes."""
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(baseline_predictions.loc[:, outcome_names], index=True).values.tobytes())
    hasher.update(json.dumps([list(X.columns), list(outcome_names)]).encode("utf-8"))
    return hasher.hexdigest()


def _checkpoint_save(path: Union[str, Path], payload: dict) -> None:
    """Atomically replace a completed-employee checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)


def _checkpoint_load(path: Union[str, Path], signature: str) -> Optional[dict]:
    target = Path(path)
    if not target.exists():
        return None
    with target.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        raise ValueError(
            f"Checkpoint {target} does not match this model/data/threshold configuration. "
            "Use a new checkpoint filename; do not mix results from different RF models."
        )
    return payload


def _finite_or_none(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def within_bounds(values: Sequence[float], lb: Sequence[float], ub: Sequence[float], atol: float = 1e-7) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower_ok = np.ones(len(values), dtype=bool)
    upper_ok = np.ones(len(values), dtype=bool)
    for j, value in enumerate(values):
        lower = _finite_or_none(lb[j])
        upper = _finite_or_none(ub[j])
        if lower is not None:
            lower_ok[j] = value >= lower - atol
        if upper is not None:
            upper_ok[j] = value <= upper + atol
    return lower_ok & upper_ok


def jointly_valid(values: Sequence[float], lb: Sequence[float], ub: Sequence[float], atol: float = 1e-7) -> bool:
    return bool(within_bounds(values, lb, ub, atol=atol).all())


def model_hyperparameters(model) -> str:
    """Return a concise JSON representation of the fitted estimator settings."""
    params = model.get_params(deep=False)
    preferred = [
        "max_depth", "min_samples_leaf", "min_samples_split", "criterion",
        "hidden_layer_sizes", "activation", "solver", "alpha", "max_iter",
        "random_state",
    ]
    compact = {key: params[key] for key in preferred if key in params}
    return json.dumps(compact, default=str, sort_keys=True)


def evaluate_predictive_adequacy(
    fitted_model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    outcome_names: Sequence[str],
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the selected model using repeated shuffled K-fold CV.

    Each training-fold outcome mean is used as a simple prediction baseline.
    The returned detail table contains pooled out-of-fold metrics for every
    repeat and the summary reports the mean and standard deviation over repeats.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    detail_rows: List[dict] = []

    for repeat in range(n_repeats):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        model_pred = np.full((len(X), len(outcome_names)), np.nan, dtype=float)
        baseline_pred = np.full_like(model_pred, np.nan)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            estimator = clone(fitted_model)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_pred = np.asarray(estimator.predict(X.iloc[test_idx]), dtype=float)
            if fold_pred.ndim == 1:
                fold_pred = fold_pred.reshape(-1, 1)
            model_pred[test_idx, :] = fold_pred
            baseline_pred[test_idx, :] = np.asarray(y.iloc[train_idx].mean(axis=0), dtype=float)

        for j, outcome in enumerate(outcome_names):
            truth = y.iloc[:, j].to_numpy(dtype=float)
            for predictor, predictions in (
                ("selected_model", model_pred[:, j]),
                ("outcome_mean_baseline", baseline_pred[:, j]),
            ):
                detail_rows.append(
                    {
                        "repeat": repeat + 1,
                        "outcome": outcome,
                        "predictor": predictor,
                        "rmse": mean_squared_error(truth, predictions) ** 0.5,
                        "mae": mean_absolute_error(truth, predictions),
                        "r2": r2_score(truth, predictions),
                        "n": len(truth),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["outcome", "predictor"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            repeats=("repeat", "nunique"),
            n=("n", "max"),
        )
    )
    summary[["rmse_sd", "mae_sd", "r2_sd"]] = summary[["rmse_sd", "mae_sd", "r2_sd"]].fillna(0.0)
    return detail, summary


def print_predictive_adequacy(summary: pd.DataFrame) -> None:
    cols = ["outcome", "predictor", "rmse_mean", "mae_mean", "r2_mean"]
    print("\nPredictive adequacy (repeated out-of-fold evaluation):\n")
    print(summary[cols].round(3).to_string(index=False))
    print()


def model_master_for_constraints(
    model_master: pd.DataFrame,
    outcome_names: Sequence[str],
    lb: Sequence[float],
    ub: Sequence[float],
    active_outcomes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    constrained = model_master.copy(deep=True)
    active = set(outcome_names if active_outcomes is None else active_outcomes)
    lower_map = dict(zip(outcome_names, lb))
    upper_map = dict(zip(outcome_names, ub))
    constrained["lb"] = [
        lower_map[name] if name in active and _finite_or_none(lower_map[name]) is not None else np.nan
        for name in constrained["outcome"]
    ]
    constrained["ub"] = [
        upper_map[name] if name in active and _finite_or_none(upper_map[name]) is not None else np.nan
        for name in constrained["outcome"]
    ]
    return constrained


def adjust_candidate(
    factual: pd.Series,
    candidate: pd.Series,
    mode: str,
    step: float,
    decimals: int,
) -> Tuple[pd.Series, pd.Series]:
    """Nudge tree-boundary solutions in the same way as the original scripts.

    Returns (display_candidate, prediction_candidate). For the relative mode,
    predictions use the tiny unrounded nudge, matching the original Skills/HRM
    implementation. For the fixed mode, the rounded candidate is also used for
    prediction, matching the original Big Five implementation.
    """
    factual = pd.to_numeric(factual, errors="coerce").astype(float)
    candidate = pd.to_numeric(candidate, errors="coerce").astype(float)
    difference = candidate - factual

    if mode == "relative":
        prediction_candidate = candidate + step * difference
        display_candidate = prediction_candidate.round(decimals)
    elif mode == "fixed":
        prediction_candidate = candidate.copy()
        changed = difference.abs() > 1e-15
        prediction_candidate.loc[changed & (difference > 0)] += step
        prediction_candidate.loc[changed & (difference < 0)] -= step
        prediction_candidate = prediction_candidate.round(decimals)
        display_candidate = prediction_candidate.copy()
    elif mode == "none":
        prediction_candidate = candidate.copy()
        display_candidate = prediction_candidate.round(decimals)
    else:
        raise ValueError(f"Unknown adjustment mode: {mode}")

    return display_candidate, prediction_candidate


def _changed_features(factual: pd.Series, solution: pd.Series, features: Sequence[str], atol: float = 1e-6) -> List[str]:
    return [feature for feature in features if abs(float(solution[feature]) - float(factual[feature])) > atol]


def _solve_and_evaluate(
    index: int,
    factual: pd.Series,
    baseline_prediction: np.ndarray,
    active_outcomes: Sequence[str],
    lb: Sequence[float],
    ub: Sequence[float],
    outcome_names: Sequence[str],
    actionable_features: Sequence[str],
    solve_fn: Callable[[int, Sequence[str], Sequence[float], Sequence[float], Optional[pd.Series]], object],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    adjustment_mode: str,
    adjustment_step: float,
    display_decimals: int,
) -> dict:
    start = time.perf_counter()
    try:
        raw_result = solve_fn(index, active_outcomes, lb, ub, factual)
        if isinstance(raw_result, tuple):
            # Existing optimizer returns (CEs, CEs_, final_model).
            ce_frame = raw_result[1]
        else:
            ce_frame = raw_result
        if ce_frame is None or len(ce_frame) < 2:
            raise RuntimeError("The optimizer did not return a counterfactual row.")

        raw_candidate = pd.to_numeric(ce_frame.iloc[1, : len(factual)], errors="coerce")
        raw_candidate.index = factual.index
        if raw_candidate.isna().any():
            raise RuntimeError("The optimizer returned non-numeric input values.")

        display_candidate, prediction_candidate = adjust_candidate(
            factual=factual,
            candidate=raw_candidate,
            mode=adjustment_mode,
            step=adjustment_step,
            decimals=display_decimals,
        )
        solution_prediction = np.asarray(
            predict_fn(pd.DataFrame([prediction_candidate], columns=factual.index)), dtype=float
        ).reshape(-1)
        elapsed = time.perf_counter() - start
        return {
            "solution_available": True,
            "solve_status": "solved",
            "solve_time_seconds": elapsed,
            "display_candidate": display_candidate,
            "prediction_candidate": prediction_candidate,
            "solution_prediction": solution_prediction,
            "error": "",
        }
    except Exception as exc:  # solver/license/infeasibility errors are retained in the output
        elapsed = time.perf_counter() - start
        return {
            "solution_available": False,
            "solve_status": "unavailable",
            "solve_time_seconds": elapsed,
            "display_candidate": factual.copy(),
            "prediction_candidate": factual.copy(),
            "solution_prediction": baseline_prediction.copy(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_ce_comparison(
    X: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    outcome_names: Sequence[str],
    lb: Sequence[float],
    ub: Sequence[float],
    actionable_features: Sequence[str],
    solve_fn: Callable[[int, Sequence[str], Sequence[float], Sequence[float], Optional[pd.Series]], object],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    adjustment_mode: str,
    adjustment_step: float,
    display_decimals: int,
    sequential_clean_starts: bool = False,
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_tag: str = "",
    resume: bool = True,
    progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, dict]]:
    """Compare single-outcome CEs, two sequential CE orders, and joint MOCE.

    Evaluation is restricted to instances outside the main joint acceptable
    region. A single-outcome stage is skipped whenever its focal outcome is
    already acceptable at the profile entering that stage.

    Sequential CE is intentionally *naive*: the second stage constrains only
    the second focal outcome. It does not explicitly preserve the outcome fixed
    in the first stage. This keeps sequential CE distinct from joint MOCE and
    allows order dependence and reversals to be observed empirically.
    """
    if len(outcome_names) != 2:
        raise ValueError("Sequential CE comparison currently requires exactly two outcomes.")

    rows: List[dict] = []
    moce_cache: Dict[int, dict] = {}
    baseline_array = baseline_predictions.loc[:, outcome_names].to_numpy(dtype=float)
    invalid_indices = [
        index for index in range(len(X)) if not jointly_valid(baseline_array[index], lb, ub)
    ]

    signature = json.dumps({
        "kind": "main_ce_comparison_v1",
        "data": _checkpoint_data_signature(X, baseline_predictions, outcome_names),
        "model": checkpoint_tag,
        "lb": list(map(float, lb)), "ub": list(map(float, ub)),
        "actionable_features": list(actionable_features),
        "adjustment_mode": adjustment_mode, "adjustment_step": adjustment_step,
        "display_decimals": display_decimals,
        "sequential_clean_starts": sequential_clean_starts,
    }, sort_keys=True)
    if checkpoint_path is not None and resume:
        saved_state = _checkpoint_load(checkpoint_path, signature)
        if saved_state is not None:
            rows = saved_state["rows"]
            moce_cache = saved_state["moce_cache"]
            completed = set(map(int, moce_cache))
            if not completed.issubset(set(invalid_indices)):
                raise ValueError("Saved checkpoint contains employees outside the current main cohort.")
            if len(rows) != 5 * len(completed):
                raise ValueError("Saved checkpoint has an inconsistent number of comparison rows.")
            if progress:
                print(f"[MAIN] Resumed {len(completed)}/{len(invalid_indices)} completed employees from {checkpoint_path}", flush=True)

    def unchanged_result(profile: pd.Series, prediction: np.ndarray) -> dict:
        profile = profile.astype(float).copy()
        return {
            "solution_available": True,
            "solve_status": "factual_already_valid",
            "solve_time_seconds": 0.0,
            "display_candidate": profile.copy(),
            "prediction_candidate": profile.copy(),
            "solution_prediction": np.asarray(prediction, dtype=float).copy(),
            "error": "",
        }

    def add_common_fields(
        row: dict,
        factual: pd.Series,
        baseline_prediction: np.ndarray,
        baseline_validity: np.ndarray,
        final_result: dict,
    ) -> Tuple[np.ndarray, List[str], float]:
        solution_prediction = np.asarray(final_result["solution_prediction"], dtype=float)
        solution_validity = within_bounds(solution_prediction, lb, ub)
        changed = _changed_features(factual, final_result["display_candidate"], actionable_features)
        l1_distance = float(
            sum(
                abs(float(final_result["display_candidate"][feature]) - float(factual[feature]))
                for feature in actionable_features
            )
        )
        row.update(
            {
                "solution_available": bool(final_result["solution_available"]),
                "solve_status": final_result["solve_status"],
                "joint_valid": bool(solution_validity.all()),
                "change_burden_l1": l1_distance,
                "changed_variable_count": len(changed),
                "changed_variables": "; ".join(changed),
                "solve_time_seconds": float(final_result["solve_time_seconds"]),
                "error": final_result["error"],
            }
        )
        for feature in factual.index:
            row[f"baseline_input_{feature}"] = float(factual[feature])
            row[f"solution_input_{feature}"] = float(final_result["display_candidate"][feature])
            row[f"change_input_{feature}"] = float(final_result["display_candidate"][feature] - factual[feature])
        for j, outcome in enumerate(outcome_names):
            row[f"baseline_{outcome}"] = float(baseline_prediction[j])
            row[f"solution_{outcome}"] = float(solution_prediction[j])
            row[f"baseline_{outcome}_valid"] = bool(baseline_validity[j])
            row[f"solution_{outcome}_valid"] = bool(solution_validity[j])
        return solution_validity, changed, l1_distance

    for position, index in enumerate(invalid_indices, start=1):
        if index in moce_cache:
            continue
        employee_start = time.perf_counter()
        if progress:
            print(f"[MAIN] {position}/{len(invalid_indices)} | employee row {index} | START", flush=True)
        factual = X.iloc[index].astype(float)
        baseline_prediction = baseline_array[index]
        baseline_validity = within_bounds(baseline_prediction, lb, ub)

        # ------------------------------------------------------------
        # Independent single-outcome CEs
        # ------------------------------------------------------------
        for focal_index in (0, 1):
            method = f"CE: {outcome_names[focal_index]}"
            if baseline_validity[focal_index]:
                result = unchanged_result(factual, baseline_prediction)
            else:
                result = _solve_and_evaluate(
                    index=index,
                    factual=factual,
                    baseline_prediction=baseline_prediction,
                    active_outcomes=[outcome_names[focal_index]],
                    lb=lb,
                    ub=ub,
                    outcome_names=outcome_names,
                    actionable_features=actionable_features,
                    solve_fn=solve_fn,
                    predict_fn=predict_fn,
                    adjustment_mode=adjustment_mode,
                    adjustment_step=adjustment_step,
                    display_decimals=display_decimals,
                )

            row = {
                "instance_index": index,
                "method": method,
                "focal_outcome": outcome_names[focal_index],
                "sequence_order": "",
                "stages_required": np.nan,
                "cumulative_path_burden_l1": np.nan,
                "stage_1_change_burden_l1": np.nan,
                "stage_2_change_burden_l1": np.nan,
                "stage_1_target_outcome": "",
                "stage_2_target_outcome": "",
                "stage_1_changed_variables": "",
                "stage_2_changed_variables": "",
            }
            solution_validity, _, _ = add_common_fields(
                row, factual, baseline_prediction, baseline_validity, result
            )
            other_index = 1 - focal_index
            row["focal_valid"] = bool(solution_validity[focal_index])
            row["other_outcome_valid"] = bool(solution_validity[other_index])
            row["other_outcome_preserved"] = (
                bool(solution_validity[other_index]) if baseline_validity[other_index] else np.nan
            )
            rows.append(row)

        # ------------------------------------------------------------
        # Sequential CEs in both orders
        # ------------------------------------------------------------
        for first_index, second_index in ((0, 1), (1, 0)):
            first_name = outcome_names[first_index]
            second_name = outcome_names[second_index]
            method = f"Sequential: {first_name} -> {second_name}"

            current_profile = factual.copy()
            current_prediction = baseline_prediction.copy()
            current_validity = baseline_validity.copy()
            stage_results: List[dict] = []
            stage_changed: List[List[str]] = []
            stage_burdens: List[float] = []
            stage_targets: List[str] = []
            errors: List[str] = []
            failed = False

            for stage_number, focal_index in enumerate((first_index, second_index), start=1):
                if current_validity[focal_index]:
                    continue

                stage_start_profile = current_profile.copy()
                stage_start_prediction = current_prediction.copy()
                stage_result = _solve_and_evaluate(
                    index=index,
                    factual=stage_start_profile,
                    baseline_prediction=stage_start_prediction,
                    active_outcomes=[outcome_names[focal_index]],
                    lb=lb,
                    ub=ub,
                    outcome_names=outcome_names,
                    actionable_features=actionable_features,
                    solve_fn=solve_fn,
                    predict_fn=predict_fn,
                    adjustment_mode=adjustment_mode,
                    adjustment_step=adjustment_step,
                    display_decimals=display_decimals,
                )
                stage_results.append(stage_result)
                stage_targets.append(outcome_names[focal_index])
                execution_stage = len(stage_results)
                if stage_result["error"]:
                    errors.append(f"stage_{execution_stage}: {stage_result['error']}")

                changed_stage = _changed_features(
                    stage_start_profile, stage_result["display_candidate"], actionable_features
                )
                burden_stage = float(
                    sum(
                        abs(
                            float(stage_result["display_candidate"][feature])
                            - float(stage_start_profile[feature])
                        )
                        for feature in actionable_features
                    )
                )
                stage_changed.append(changed_stage)
                stage_burdens.append(burden_stage)

                if not stage_result["solution_available"]:
                    failed = True
                    break

                # For Skills, pass the clean display values to the next stage;
                # other settings retain the adjusted prediction candidate.
                next_profile_key = (
                    "display_candidate" if sequential_clean_starts
                    else "prediction_candidate"
                )
                current_profile = stage_result[next_profile_key].astype(float).copy()
                current_prediction = np.asarray(stage_result["solution_prediction"], dtype=float).copy()
                current_validity = within_bounds(current_prediction, lb, ub)

            stages_required = len(stage_results)
            total_solve_time = float(sum(r["solve_time_seconds"] for r in stage_results))

            if stage_results:
                last = stage_results[-1]
                final_display = last["display_candidate"].astype(float).copy()
                final_prediction_profile = last["prediction_candidate"].astype(float).copy()
                final_prediction = np.asarray(last["solution_prediction"], dtype=float).copy()
                if sequential_clean_starts:
                    _, final_prediction_profile = adjust_candidate(
                        factual=factual,
                        candidate=final_display,
                        mode=adjustment_mode,
                        step=adjustment_step,
                        decimals=display_decimals,
                    )
                    final_prediction = np.asarray(
                        predict_fn(
                            pd.DataFrame(
                                [final_prediction_profile],
                                columns=factual.index,
                            )
                        ),
                        dtype=float,
                    ).reshape(-1)
            else:
                final_display = factual.copy()
                final_prediction_profile = factual.copy()
                final_prediction = baseline_prediction.copy()

            if failed:
                sequential_status = f"stage_{stages_required}_unavailable"
            else:
                sequential_status = f"solved_{stages_required}_stage" + ("s" if stages_required != 1 else "")

            final_result = {
                "solution_available": not failed,
                "solve_status": sequential_status,
                "solve_time_seconds": total_solve_time,
                "display_candidate": final_display,
                "prediction_candidate": final_prediction_profile,
                "solution_prediction": final_prediction,
                "error": " | ".join(errors),
            }
            row = {
                "instance_index": index,
                "method": method,
                "focal_outcome": "sequential",
                "sequence_order": f"{first_name} -> {second_name}",
                "stages_required": stages_required,
                "cumulative_path_burden_l1": float(sum(stage_burdens)),
                "stage_1_change_burden_l1": stage_burdens[0] if len(stage_burdens) >= 1 else 0.0,
                "stage_2_change_burden_l1": stage_burdens[1] if len(stage_burdens) >= 2 else 0.0,
                "stage_1_target_outcome": stage_targets[0] if len(stage_targets) >= 1 else "",
                "stage_2_target_outcome": stage_targets[1] if len(stage_targets) >= 2 else "",
                "stage_1_changed_variables": "; ".join(stage_changed[0]) if len(stage_changed) >= 1 else "",
                "stage_2_changed_variables": "; ".join(stage_changed[1]) if len(stage_changed) >= 2 else "",
            }
            solution_validity, _, _ = add_common_fields(
                row, factual, baseline_prediction, baseline_validity, final_result
            )
            # Focal/other-outcome terminology is not meaningful for a completed
            # two-stage sequence, so joint validity is the relevant comparison.
            row["focal_valid"] = np.nan
            row["other_outcome_valid"] = np.nan
            row["other_outcome_preserved"] = np.nan
            for stage_no, stage_result in enumerate(stage_results, start=1):
                stage_pred = np.asarray(stage_result["solution_prediction"], dtype=float)
                for j, outcome in enumerate(outcome_names):
                    row[f"stage_{stage_no}_{outcome}"] = float(stage_pred[j])
                    row[f"stage_{stage_no}_{outcome}_valid"] = bool(
                        within_bounds(stage_pred, lb, ub)[j]
                    )
            rows.append(row)

        # ------------------------------------------------------------
        # Joint MOCE
        # ------------------------------------------------------------
        result = _solve_and_evaluate(
            index=index,
            factual=factual,
            baseline_prediction=baseline_prediction,
            active_outcomes=list(outcome_names),
            lb=lb,
            ub=ub,
            outcome_names=outcome_names,
            actionable_features=actionable_features,
            solve_fn=solve_fn,
            predict_fn=predict_fn,
            adjustment_mode=adjustment_mode,
            adjustment_step=adjustment_step,
            display_decimals=display_decimals,
        )
        row = {
            "instance_index": index,
            "method": "MOCE",
            "focal_outcome": "joint",
            "sequence_order": "",
            "stages_required": np.nan,
            "cumulative_path_burden_l1": np.nan,
            "stage_1_change_burden_l1": np.nan,
            "stage_2_change_burden_l1": np.nan,
            "stage_1_changed_variables": "",
            "stage_2_changed_variables": "",
        }
        solution_validity, changed, l1_distance = add_common_fields(
            row, factual, baseline_prediction, baseline_validity, result
        )
        row["focal_valid"] = bool(solution_validity.all())
        row["other_outcome_valid"] = np.nan
        row["other_outcome_preserved"] = np.nan
        rows.append(row)

        cached_result = dict(result)
        cached_result.update(
            {
                "baseline_prediction": baseline_prediction.copy(),
                "solution_validity": solution_validity.copy(),
                "changed_features": changed,
                "change_burden_l1": l1_distance,
            }
        )
        moce_cache[index] = cached_result

        if checkpoint_path is not None:
            _checkpoint_save(checkpoint_path, {
                "signature": signature, "rows": rows, "moce_cache": moce_cache,
            })
            # Readable, partial comparison table. Only the .pkl is used to resume.
            pd.DataFrame(rows).to_csv(Path(checkpoint_path).with_suffix(".csv"), index=False)
        if progress:
            elapsed = time.perf_counter() - employee_start
            status = "valid" if bool(result["solution_available"]) and bool(solution_validity.all()) else "unavailable/invalid"
            print(
                f"[MAIN] {position}/{len(invalid_indices)} | employee row {index} | DONE "
                f"| {elapsed:.1f}s | MOCE {status} | saved {len(moce_cache)}/{len(invalid_indices)}",
                flush=True,
            )

    instances = pd.DataFrame(rows)
    summary = summarize_ce_comparison(instances)
    return instances, summary, moce_cache

def summarize_ce_comparison(instances: pd.DataFrame) -> pd.DataFrame:
    def bool_rate(series: pd.Series) -> float:
        defined = series.dropna()
        if len(defined) == 0:
            return np.nan
        return float(defined.astype(bool).mean())

    rows: List[dict] = []
    for method, group in instances.groupby("method", sort=False):
        available = group["solution_available"].astype(bool)
        available_group = group.loc[available]
        sequential = group["sequence_order"].fillna("").astype(str).str.len() > 0
        sequential_group = group.loc[sequential]
        rows.append(
            {
                "method": method,
                "n_instances": len(group),
                "solution_available_n": int(available.sum()),
                "solution_available_rate": float(available.mean()) if len(group) else np.nan,
                "focal_valid_rate": bool_rate(group["focal_valid"]),
                "joint_valid_rate": bool_rate(group["joint_valid"]),
                "other_outcome_valid_rate": bool_rate(group["other_outcome_valid"]),
                "other_outcome_preserved_rate": bool_rate(group["other_outcome_preserved"]),
                "mean_change_burden_l1": float(available_group["change_burden_l1"].mean()) if len(available_group) else np.nan,
                "median_change_burden_l1": float(available_group["change_burden_l1"].median()) if len(available_group) else np.nan,
                "mean_changed_variables": float(available_group["changed_variable_count"].mean()) if len(available_group) else np.nan,
                "median_changed_variables": float(available_group["changed_variable_count"].median()) if len(available_group) else np.nan,
                "mean_solve_time_seconds": float(group["solve_time_seconds"].mean()) if len(group) else np.nan,
                "median_solve_time_seconds": float(group["solve_time_seconds"].median()) if len(group) else np.nan,
                "mean_stages_required": float(sequential_group["stages_required"].mean()) if len(sequential_group) else np.nan,
                "median_stages_required": float(sequential_group["stages_required"].median()) if len(sequential_group) else np.nan,
                "mean_cumulative_path_burden_l1": float(sequential_group.loc[sequential_group["solution_available"].astype(bool), "cumulative_path_burden_l1"].mean()) if len(sequential_group) else np.nan,
                "median_cumulative_path_burden_l1": float(sequential_group.loc[sequential_group["solution_available"].astype(bool), "cumulative_path_burden_l1"].median()) if len(sequential_group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def print_ce_comparison(summary: pd.DataFrame) -> None:
    cols = [
        "method", "solution_available_rate", "joint_valid_rate",
        "mean_change_burden_l1", "mean_changed_variables", "median_solve_time_seconds",
    ]
    if "mean_stages_required" in summary.columns:
        cols.append("mean_stages_required")
    print("\nSingle-outcome CE, sequential CE, and MOCE comparison:\n")
    print(summary[cols].round(3).to_string(index=False))
    print()

def evaluate_threshold_sensitivity(
    threshold_specs: Mapping[str, Mapping[str, Sequence[float]]],
    main_name: str,
    X: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    outcome_names: Sequence[str],
    actionable_features: Sequence[str],
    solve_fn: Callable[[int, Sequence[str], Sequence[float], Sequence[float], Optional[pd.Series]], object],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    adjustment_mode: str,
    adjustment_step: float,
    display_decimals: int,
    main_moce_cache: Optional[Mapping[int, dict]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_tag: str = "",
    resume: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    rows: List[dict] = []
    predictions = baseline_predictions.loc[:, outcome_names].to_numpy(dtype=float)
    signature = json.dumps({
        "kind": "threshold_sensitivity_v1",
        "data": _checkpoint_data_signature(X, baseline_predictions, outcome_names),
        "model": checkpoint_tag,
        "threshold_specs": {
            name: {"lb": list(map(float, bounds["lb"])), "ub": list(map(float, bounds["ub"]))}
            for name, bounds in threshold_specs.items()
        },
        "main_name": main_name,
        "actionable_features": list(actionable_features),
        "adjustment_mode": adjustment_mode, "adjustment_step": adjustment_step,
        "display_decimals": display_decimals,
    }, sort_keys=True)
    records_by_spec = {}
    if checkpoint_path is not None and resume:
        saved_state = _checkpoint_load(checkpoint_path, signature)
        if saved_state is not None:
            records_by_spec = saved_state["records_by_spec"]
            if progress:
                n_saved = sum(map(len, records_by_spec.values()))
                print(f"[SENSITIVITY] Resumed {n_saved} employee-scenario records from {checkpoint_path}", flush=True)

    for specification, bounds in threshold_specs.items():
        lb = list(bounds["lb"])
        ub = list(bounds["ub"])
        initial_valid = np.array([jointly_valid(pred, lb, ub) for pred in predictions], dtype=bool)
        requiring_change = np.where(~initial_valid)[0]
        records: List[dict] = []
        saved_records = records_by_spec.setdefault(specification, {})
        if not set(map(int, saved_records)).issubset(set(map(int, requiring_change))):
            raise ValueError(f"Sensitivity checkpoint has unexpected indices for {specification}.")
        if progress:
            print(
                f"[SENSITIVITY {specification}] {len(saved_records)}/{len(requiring_change)} employees already completed",
                flush=True,
            )

        for position, index in enumerate(requiring_change, start=1):
            index = int(index)
            if index in saved_records:
                records.append(saved_records[index])
                continue
            employee_start = time.perf_counter()
            if progress:
                print(f"[SENSITIVITY {specification}] {position}/{len(requiring_change)} | employee row {index} | START", flush=True)
            factual = X.iloc[index].astype(float)
            baseline_prediction = predictions[index]
            if specification == main_name and main_moce_cache is not None and index in main_moce_cache:
                result = dict(main_moce_cache[index])
            else:
                result = _solve_and_evaluate(
                    index=index,
                    factual=factual,
                    baseline_prediction=baseline_prediction,
                    active_outcomes=list(outcome_names),
                    lb=lb,
                    ub=ub,
                    outcome_names=outcome_names,
                    actionable_features=actionable_features,
                    solve_fn=solve_fn,
                    predict_fn=predict_fn,
                    adjustment_mode=adjustment_mode,
                    adjustment_step=adjustment_step,
                    display_decimals=display_decimals,
                )

            solution_prediction = np.asarray(result["solution_prediction"], dtype=float)
            display_candidate = result["display_candidate"]
            changed = _changed_features(factual, display_candidate, actionable_features)
            distance = float(sum(abs(float(display_candidate[f]) - float(factual[f])) for f in actionable_features))
            record = {
                "solution_available": bool(result["solution_available"]),
                "joint_valid": jointly_valid(solution_prediction, lb, ub),
                "distance": distance,
                "changed_count": len(changed),
                "solve_time": float(result["solve_time_seconds"]),
            }
            records.append(record)
            saved_records[index] = record
            if checkpoint_path is not None:
                _checkpoint_save(checkpoint_path, {
                    "signature": signature, "records_by_spec": records_by_spec,
                })
            if progress:
                elapsed = time.perf_counter() - employee_start
                status = "valid" if record["solution_available"] and record["joint_valid"] else "unavailable/invalid"
                print(
                    f"[SENSITIVITY {specification}] {position}/{len(requiring_change)} | employee row {index} "
                    f"| DONE | {elapsed:.1f}s | MOCE {status}", flush=True,
                )

        record_df = pd.DataFrame(records)
        available = record_df["solution_available"].astype(bool) if len(record_df) else pd.Series(dtype=bool)
        valid = record_df["joint_valid"].astype(bool) if len(record_df) else pd.Series(dtype=bool)
        available_df = record_df.loc[available] if len(record_df) else record_df
        row = {
            "threshold_specification": specification,
            "n_total": len(X),
            "n_initially_joint_valid": int(initial_valid.sum()),
            "initial_joint_valid_rate": float(initial_valid.mean()),
            "n_requiring_change": int(len(requiring_change)),
            "solution_available_n": int(available.sum()) if len(record_df) else 0,
            "solution_available_rate": float(available.mean()) if len(record_df) else 1.0,
            "joint_valid_n": int(valid.sum()) if len(record_df) else 0,
            "joint_valid_rate": float(valid.mean()) if len(record_df) else 1.0,
            "mean_change_burden_l1": float(available_df["distance"].mean()) if len(available_df) else 0.0,
            "median_change_burden_l1": float(available_df["distance"].median()) if len(available_df) else 0.0,
            "mean_changed_variables": float(available_df["changed_count"].mean()) if len(available_df) else 0.0,
            "median_changed_variables": float(available_df["changed_count"].median()) if len(available_df) else 0.0,
            "mean_solve_time_seconds": float(record_df["solve_time"].mean()) if len(record_df) else 0.0,
            "median_solve_time_seconds": float(record_df["solve_time"].median()) if len(record_df) else 0.0,
        }
        for j, outcome in enumerate(outcome_names):
            row[f"{outcome}_lb"] = lb[j]
            row[f"{outcome}_ub"] = ub[j]
        rows.append(row)

    return pd.DataFrame(rows)


def print_threshold_sensitivity(summary: pd.DataFrame) -> None:
    cols = [
        "threshold_specification", "initial_joint_valid_rate", "solution_available_rate",
        "joint_valid_rate", "mean_change_burden_l1", "mean_changed_variables",
    ]
    print("\nThreshold sensitivity:\n")
    print(summary[cols].round(3).to_string(index=False))
    print()


def build_moce_frames(
    X: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    outcome_names: Sequence[str],
    lb: Sequence[float],
    ub: Sequence[float],
    moce_cache: Mapping[int, dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]:
    """Recreate the original baseline/MOCE frames while preserving its row order.

    As in the original scripts, employees requiring recourse are placed first,
    followed by employees already inside the acceptable region.
    """
    predictions = baseline_predictions.loc[:, outcome_names].to_numpy(dtype=float)
    invalid = [index for index in range(len(X)) if not jointly_valid(predictions[index], lb, ub)]
    valid = [index for index in range(len(X)) if jointly_valid(predictions[index], lb, ub)]
    ordered_indices = invalid + valid

    baseline_rows: List[pd.Series] = []
    solution_rows: List[pd.Series] = []
    for index in ordered_indices:
        factual = X.iloc[index].astype(float)
        baseline_row = pd.concat(
            [factual, pd.Series(predictions[index], index=outcome_names, dtype=float)]
        )
        baseline_rows.append(baseline_row)

        if index in moce_cache and moce_cache[index]["solution_available"]:
            solution_inputs = moce_cache[index]["display_candidate"].astype(float)
            solution_prediction = np.asarray(moce_cache[index]["solution_prediction"], dtype=float)
            solution_row = pd.concat(
                [solution_inputs, pd.Series(solution_prediction, index=outcome_names, dtype=float)]
            )
        else:
            solution_row = baseline_row.copy()
        solution_rows.append(solution_row)

    DF_0 = pd.DataFrame(baseline_rows).reset_index(drop=True)
    DF_1 = pd.DataFrame(solution_rows).reset_index(drop=True)
    DF_diff = DF_1 - DF_0
    return DF_0, DF_1, DF_diff, ordered_indices


def selected_case_comparison(
    comparison_instances: pd.DataFrame,
    selected_display_indices: Sequence[int],
    ordered_original_indices: Sequence[int],
) -> pd.DataFrame:
    original_indices = [ordered_original_indices[i] for i in selected_display_indices if 0 <= i < len(ordered_original_indices)]
    selected = comparison_instances[comparison_instances["instance_index"].isin(original_indices)].copy()
    display_map = {original: display for display, original in enumerate(ordered_original_indices)}
    selected.insert(1, "display_index", selected["instance_index"].map(display_map))
    return selected.sort_values(["display_index", "method"])


def save_setting_formulation(
    output_path: Union[str, Path],
    setting: str,
    purpose: str,
    actionable_features: Sequence[str],
    fixed_features: Sequence[str],
    variable_role: str,
    outcome_names: Sequence[str],
    main_lb: Sequence[float],
    main_ub: Sequence[float],
    feasibility_restrictions: str,
    distance: str,
    interpretation: str,
    model_name: str,
    hyperparameters: str,
) -> pd.DataFrame:
    row = {
        "setting": setting,
        "decision_purpose": purpose,
        "variables_allowed_to_change_or_vary": "; ".join(actionable_features),
        "variable_role": variable_role,
        "contextual_or_fixed_variables": "; ".join(fixed_features),
        "outcomes": "; ".join(outcome_names),
        "main_acceptable_region": "; ".join(
            f"{name}: [{main_lb[j]}, {main_ub[j]}]" for j, name in enumerate(outcome_names)
        ),
        "feasibility_restrictions": feasibility_restrictions,
        "distance": distance,
        "interpretation": interpretation,
        "predictive_model": model_name,
        "hyperparameters": hyperparameters,
    }
    frame = pd.DataFrame([row])
    frame.to_csv(output_path, index=False)
    return frame
