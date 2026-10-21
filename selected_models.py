"""Selected multi-output Random Forest models and optimization exports.

The configurations were selected using 3x5-fold cross-validation in
compare_predictive_models.py (not included in this minimal execution package).
Training uses all observations and preserves the selected RF parameters.
"""
from __future__ import annotations
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import constraint_learning as cl

SELECTED = {
    "skills": {"model": "rf", "name": "RF_depth4_leaf6",
               "params": {"n_estimators": 150, "max_depth": 4,
                          "min_samples_leaf": 6, "max_features": 1.0,
                          "random_state": 42, "n_jobs": 1}},
    "big_five": {"model": "rf", "name": "RF_depth4_leaf6",
                 "params": {"n_estimators": 150, "max_depth": 4,
                            "min_samples_leaf": 6, "max_features": 1.0,
                            "random_state": 42, "n_jobs": 1}},
    "hrm_practices": {"model": "rf", "name": "RF_depth6_leaf3",
                      "params": {"n_estimators": 150, "max_depth": 6,
                                 "min_samples_leaf": 3, "max_features": 1.0,
                                 "random_state": 42, "n_jobs": 1}},
}


def train_and_export(setting: str, X: pd.DataFrame, y: pd.DataFrame,
                     output_root: str = "results"):
    """Fit the selected multi-output RF and export Pyomo-ready leaf constraints.

    Both outcome tables retain the same tree partitions, with each tree's
    prediction for the corresponding outcome. Models are fitted jointly,
    not re-fitted separately for each outcome.
    """
    spec = SELECTED[setting]
    model_type = spec["model"]
    if model_type != "rf":
        raise ValueError(f"This package expects Random Forest for {setting}.")
    fitted = RandomForestRegressor(**spec["params"])
    fitted.fit(X, y)
    embed_source = fitted

    target_dir = Path(output_root) / model_type
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for j, outcome in enumerate(y.columns):
        learned = cl.ConstraintLearning(X, y.iloc[:, j], embed_source, model_type, j)
        table = learned.constraint_extrapolation("continuous")
        path = target_dir / f"V1_{outcome}_model.csv"
        table.to_csv(path, index=False)
        paths[outcome] = str(path)
        if model_type == "rf":
            # Verify each exported leaf value reproduces the fitted RF output.
            leaf_predictions = table[["Tree_id", "ID", "prediction"]].drop_duplicates()
            reconstructed = np.zeros(len(X), dtype=float)
            for tree_id, tree in enumerate(embed_source.estimators_):
                leaf_nodes = np.flatnonzero(tree.tree_.children_left == -1)
                labels = {int(node): i + 1 for i, node in enumerate(leaf_nodes)}
                values = leaf_predictions.loc[leaf_predictions["Tree_id"] == tree_id]
                values = values.set_index("ID")["prediction"]
                reconstructed += np.array([
                    values.loc[labels[int(node)]] for node in tree.apply(X)
                ], dtype=float) / len(embed_source.estimators_)
            error = float(np.max(np.abs(reconstructed - fitted.predict(X)[:, j])))
            if error > 1e-7:
                raise AssertionError(f"RF output {outcome} table mismatch: {error}")

    model_dir = Path(output_root) / "selected_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, model_dir / f"{setting}_{spec['name']}.joblib")
    metadata = {"setting": setting, "model_name": spec["name"],
                "model_type": model_type, "parameters": spec["params"],
                "feature_columns": list(X.columns), "outcomes": list(y.columns),
                "trained_n": len(X), "embedding_tables": paths}
    (model_dir / f"{setting}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return fitted, model_type, paths, spec["name"]
