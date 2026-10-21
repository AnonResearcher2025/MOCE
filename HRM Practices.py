# ============================================================
# HRM MOCE pipeline
# ============================================================
# Purpose:
#   End-to-end HRM MOCE pipeline used in the paper.
# Steps:
#   1) Load Data-HRM.csv
#   2) Fit the multi-outcome model and assess predictive adequacy
#   3) Compare single-outcome CEs with the joint MOCE
#   4) Evaluate lenient/main/demanding outcome thresholds
#   5) Construct DF_0 (baseline), DF_1 (MOCE), and DF_diff dataframes
#   6) Print illustrative employees and aggregate change patterns
#   7) Plot the original compact 4×4 histogram grid
# ============================================================

import io
import random
import contextlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

import ce_helpers as ch
from selected_models import train_and_export
import moce_evaluation as me

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# User settings
# ============================================================
t = 4
o = 2
rn = 42

# Chosen after optimization from the new model; not the old paper's cases.
selected_idx = []

# MOCE settings
num_counterfactuals = 1
model = "rf"
mu = 0
obj = "l1"

# Empirical evaluation settings
cv_folds = 5
cv_repeats = 3
main_threshold_name = "main"
evaluation_dir = "results/evaluation"
setting_slug = "hrm_practices"

# Actionable inputs (allowed to change)
M = [
    "Prehire Tests", "Structured Interviews", "Employee Participation",
    "Complaint Procedure", "Group Bonuses", "Individual Bonuses",
    "Performance Evaluation", "Goals Communication", "Suggestion Adoption",
    "Performance Pay", "Internal Promotion", "Job Autonomy",
    "Selective Hiring", "Pay Competitiveness", "Job Training",
]
# Inputs may increase or decrease
L = []  

# Success bounds
lb = [6, 1]
ub = [7, 3]

# Lenient/main/demanding outcome requirements
threshold_specs = {
    "lenient": {"lb": [5.9, 1], "ub": [7, 3.1]},
    "main": {"lb": lb, "ub": ub},
    "demanding": {"lb": [6.1, 1], "ub": [7, 2.9]},
}


# ============================================================
# Functions to print the tables
# ============================================================
def print_selected_employees_compact(
    DF_0: pd.DataFrame,
    DF_1: pd.DataFrame,
    selected_idx: list,
    out_cols: list,
    decimals_inputs: int = 2,
    decimals_outcomes: int = 2,
    atol: float = 1e-9,
):
    missing = [i for i in selected_idx if i not in DF_0.index or i not in DF_1.index]
    if missing:
        raise KeyError(f"Selected indices not found in DF_0/DF_1: {missing}")

    row_order_inputs = [c for c in DF_0.columns if c not in out_cols]
    row_order = row_order_inputs + list(out_cols)

    def _fmt(v, d):
        if pd.isna(v):
            return ""
        return f"{float(v):.{d}f}"

    def _changed(v0, v1):
        if pd.isna(v0) and pd.isna(v1):
            return False
        if pd.isna(v0) != pd.isna(v1):
            return True
        return abs(float(v0) - float(v1)) > atol

    display_df = pd.DataFrame(index=row_order)

    for j, idx in enumerate(selected_idx, start=1):
        col_vals = []
        for var in row_order:
            v0 = DF_0.loc[idx, var]
            v1 = DF_1.loc[idx, var]

            if var in out_cols:
                s0 = _fmt(v0, decimals_outcomes)
                s1 = _fmt(v1, decimals_outcomes)
                cell = f"{s0} ({s1})" if (s0 or s1) else ""
            else:
                s0 = _fmt(v0, decimals_inputs)
                s1 = _fmt(v1, decimals_inputs)
                if not s0 and not s1:
                    cell = ""
                else:
                    cell = f"{s0} ({s1})" if _changed(v0, v1) else s0

            col_vals.append(cell)

        display_df[f"Employee {j}"] = col_vals

    print("\nSelected employees:\n")
    print(display_df.to_string())
    print("\n")


def summarize_subset_patterns_solo_then_combos(DF_diff: pd.DataFrame, subsets: dict, actionable_features: list):
    feats_in_order = [c for c in DF_diff.columns if c in actionable_features]
    if not feats_in_order:
        raise ValueError("No actionable features found in DF_diff columns.")

    out = {}
    for subset_name, mask in subsets.items():
        sub = DF_diff.loc[mask, feats_in_order].fillna(0)

        patterns = Counter()
        for _, row in sub.iterrows():
            changed = tuple([f for f in feats_in_order if row[f] != 0])
            patterns[changed] += 1

        items = []
        for pat, cnt in patterns.items():
            label = "no change" if len(pat) == 0 else " + ".join(pat)
            items.append((len(pat), -cnt, label, cnt))

        items.sort(key=lambda x: (x[0] if x[0] > 0 else 999, x[1], x[2]))
        out[subset_name] = [(label, cnt) for _k, _neg, label, cnt in items]

    return out


def print_subset_patterns(summary: dict, subsets: dict):
    print("Subset-level change patterns:")
    for subset_name, patterns in summary.items():
        n = int(subsets[subset_name].sum())
        non_nochange = [(lab, cnt) for lab, cnt in patterns if lab != "no change" and cnt > 0]
        if not non_nochange:
            print(f"- {subset_name} (n={n}): no change")
        else:
            txt = ", ".join([f"{lab} ({cnt})" for lab, cnt in non_nochange])
            print(f"- {subset_name} (n={n}): {txt}")
    print("\n")

# ============================================================
# Reproducibility
# ============================================================
random.seed(rn)
np.random.seed(rn)
_ = check_random_state(rn)


# ============================================================
# Data
# ============================================================
data = pd.read_csv("Data-HRM.csv")
out = data.columns[-o:].tolist()
n = len(data.columns) - o
X = data.iloc[:, :n]
y = data[out]


# ============================================================
# Train predictive models
# ============================================================
d = {"target": "Outcomes", "numerical": data.columns[:n].tolist()}
I = [val for val in d["numerical"] if val not in M]

models = {}
models["Outcomes"], alg, exported_paths, selected_name = train_and_export(
    setting="hrm_practices", X=X, y=y,
)
y_pred_df = pd.DataFrame(models["Outcomes"].predict(X), columns=out)
print(f"Selected model: {selected_name}")

# ============================================================
# Predictive adequacy: repeated CV versus outcome-mean baseline
# ============================================================
evaluation_path = me.ensure_output_dir(evaluation_dir)
cv_detail, cv_summary = me.evaluate_predictive_adequacy(
    fitted_model=models["Outcomes"],
    X=X,
    y=y,
    outcome_names=out,
    n_splits=cv_folds,
    n_repeats=cv_repeats,
    random_state=rn,
)
cv_detail.to_csv(evaluation_path / f"{setting_slug}_predictive_adequacy_repeats.csv", index=False)
cv_summary.to_csv(evaluation_path / f"{setting_slug}_predictive_adequacy_summary.csv", index=False)
me.print_predictive_adequacy(cv_summary)


# ============================================================
# Model registry for the optimizer
# ============================================================
model_master = pd.DataFrame(columns=["outcome", "model_type", "save_path", "task", "objective"])
for i_out in range(len(out)):
    model_master.loc[len(model_master)] = [
        out[i_out],
        alg,
        exported_paths[out[i_out]],
        "continuous",
        0,
    ]
model_master["lb"] = lb
model_master["ub"] = ub
model_master["SCM_counterfactuals"] = None
model_master["features"] = [[col for col in X.columns]] * len(model_master)


# ============================================================
# Single-outcome CE, sequential CE, and MOCE evaluation
# ============================================================
_silence2 = io.StringIO()
bounds = {name: (1, 7) for name in M}

def solve_counterfactual_for_index(ind, active_outcomes, lower_bounds, upper_bounds, factual_profile=None):
    constrained_model_master = me.model_master_for_constraints(
        model_master=model_master,
        outcome_names=out,
        lb=lower_bounds,
        ub=upper_bounds,
        active_outcomes=active_outcomes,
    )
    factual_profile = X.iloc[ind] if factual_profile is None else pd.Series(factual_profile, index=X.columns, dtype=float)
    with contextlib.redirect_stdout(_silence2):
        return ch.opt(
            X, X, factual_profile,
            [x for x in d["numerical"] if x not in M],
            [],
            M,
            {},
            I, L, [], X.columns,
            True, mu, False, False,
            num_counterfactuals,
            constrained_model_master, None,
            obj=obj,
            bounds=bounds,
        )


comparison_instances, comparison_summary, main_moce_cache = me.evaluate_ce_comparison(
    X=X,
    baseline_predictions=y_pred_df,
    outcome_names=out,
    lb=lb,
    ub=ub,
    actionable_features=M,
    solve_fn=solve_counterfactual_for_index,
    predict_fn=models["Outcomes"].predict,
    adjustment_mode="none",
    adjustment_step=0.0,
    display_decimals=3,
    checkpoint_path=evaluation_path / f"{setting_slug}_{selected_name}_main_checkpoint.pkl",
    checkpoint_tag=selected_name,
    progress=True,
)
# Report focal-outcome CEs only for employees who initially needed that CE.
# Retain all initially joint-invalid employees for sequential and MOCE methods.
ce_not_needed = pd.Series(False, index=comparison_instances.index)
for outcome_name in out:
    ce_not_needed |= (
        comparison_instances["method"].eq(f"CE: {outcome_name}")
        & comparison_instances[f"baseline_{outcome_name}_valid"].astype(bool)
    )
comparison_summary = me.summarize_ce_comparison(
    comparison_instances.loc[~ce_not_needed].copy()
)

comparison_instances.to_csv(evaluation_path / f"{setting_slug}_ce_comparison_instances.csv", index=False)
comparison_summary.to_csv(evaluation_path / f"{setting_slug}_ce_comparison_summary.csv", index=False)


# ============================================================
# Reconstruct the original MOCE baseline/solution frames
# ============================================================
DF_0, DF_1, DF_diff, ordered_original_indices = me.build_moce_frames(
    X=X,
    baseline_predictions=y_pred_df,
    outcome_names=out,
    lb=lb,
    ub=ub,
    moce_cache=main_moce_cache,
)

# Three manually selected employees (0-based dataset row IDs)
selected_original_idx = [43, 219, 86]

selected_idx = [
    ordered_original_indices.index(i)
    for i in selected_original_idx
]

# ============================================================
# Threshold sensitivity and setting formulation
# ============================================================
threshold_summary = me.evaluate_threshold_sensitivity(
    threshold_specs=threshold_specs,
    main_name=main_threshold_name,
    X=X,
    baseline_predictions=y_pred_df,
    outcome_names=out,
    actionable_features=M,
    solve_fn=solve_counterfactual_for_index,
    predict_fn=models["Outcomes"].predict,
    adjustment_mode="none",
    adjustment_step=0.0,
    display_decimals=3,
    main_moce_cache=main_moce_cache,
    checkpoint_path=evaluation_path / f"{setting_slug}_{selected_name}_sensitivity_checkpoint.pkl",
    checkpoint_tag=selected_name,
    progress=True,
)
threshold_summary.to_csv(evaluation_path / f"{setting_slug}_threshold_sensitivity.csv", index=False)

me.save_setting_formulation(
    output_path=evaluation_path / f"{setting_slug}_setting_formulation.csv",
    setting="HRM Practices",
    purpose="Organization-controlled intervention design in a higher-dimensional HRM practice space.",
    actionable_features=M,
    fixed_features=I,
    variable_role="Organization-controlled decision variables.",
    outcome_names=out,
    main_lb=lb,
    main_ub=ub,
    feasibility_restrictions="Only HRM practices may change; employee characteristics are fixed; each HRM practice is bounded between 1 and 7.",
    distance="L1 change burden over the variables allowed to change.",
    interpretation="Organization-level intervention design generated by the fitted model, not a causal estimate of practice effects.",
    model_name=selected_name,
    hyperparameters=me.model_hyperparameters(models["Outcomes"]),
)


# ============================================================
# Reorder columns: actionable first, by frequency of change
# ============================================================
columns_of_interest = [c for c in DF_diff.columns if c in M]
change_counts = {col: int((DF_diff[col] != 0).sum()) for col in columns_of_interest}
sorted_cols = sorted(change_counts, key=lambda x: change_counts[x], reverse=True)

remaining_cols = [c for c in DF_0.columns if c not in sorted_cols]
DF_0 = DF_0[sorted_cols + remaining_cols]
DF_1 = DF_1[sorted_cols + remaining_cols]
DF_diff = DF_diff[sorted_cols + remaining_cols]


# ============================================================
# Define outcome-based subsets
# ============================================================
subset1_mask = (
    (DF_0[out[0]] >= lb[0]) & (DF_0[out[0]] <= ub[0]) &
    (DF_0[out[1]] >= lb[1]) & (DF_0[out[1]] <= ub[1])
)
subset2_mask = (
    ((DF_0[out[0]] < lb[0]) | (DF_0[out[0]] > ub[0])) &
    ((DF_0[out[1]] >= lb[1]) & (DF_0[out[1]] <= ub[1]))
)
subset3_mask = (
    ((DF_0[out[0]] >= lb[0]) & (DF_0[out[0]] <= ub[0])) &
    ((DF_0[out[1]] < lb[1]) | (DF_0[out[1]] > ub[1]))
)
subset4_mask = (
    ((DF_0[out[0]] < lb[0]) | (DF_0[out[0]] > ub[0])) &
    ((DF_0[out[1]] < lb[1]) | (DF_0[out[1]] > ub[1]))
)

subsets = {
    "High Performance Low Turnover Int.": subset1_mask,
    "Low Performance Low Turnover Int.": subset2_mask,
    "High Performance High Turnover Int.": subset3_mask,
    "Low Performance High Turnover Int.": subset4_mask,
}


# ============================================================
# Prints
# ============================================================
print("\nNumber of evaluated employees per method (focal-valid CE cases only):")
print(comparison_summary[["method", "n_instances"]].to_string(index=False))
me.print_ce_comparison(comparison_summary)
me.print_threshold_sensitivity(threshold_summary)

selected_comparison = me.selected_case_comparison(
    comparison_instances=comparison_instances,
    selected_display_indices=selected_idx,
    ordered_original_indices=ordered_original_indices,
)
selected_comparison.insert(
    2, "employee_label",
    selected_comparison["instance_index"].map(
        {i: f"Employee {j}" for j, i in enumerate(selected_original_idx, start=1)}
    ),
)
selected_comparison["case_order"] = selected_comparison["instance_index"].map(
    {i: j for j, i in enumerate(selected_original_idx, start=1)}
)
selected_comparison = selected_comparison.sort_values(
    ["case_order", "method"]
).drop(columns="case_order")
selected_comparison.to_csv(evaluation_path / f"{setting_slug}_selected_case_comparison.csv", index=False)

print_selected_employees_compact(DF_0, DF_1, selected_idx, out)
subset_summary = summarize_subset_patterns_solo_then_combos(DF_diff, subsets, M)
print_subset_patterns(subset_summary, subsets)


# ============================================================
# Plotting the figure
# ============================================================
manual_outcome_bins = {
    out[0]: np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0], dtype=float),
    out[1]: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float),
}
manual_hrm_bins = {
    name: np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype=float)
    for name in M
}

def build_bins_for_column(col: str, full0: pd.DataFrame, full1: pd.DataFrame, out_cols: list):
    if col in manual_outcome_bins:
        edges = manual_outcome_bins[col]
        centers = (edges[:-1] + edges[1:]) / 2.0
        labels = [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]
        def transform(s):
            s = s.astype(float)
            eps = 1e-6
            return np.clip(s, edges[0] + eps, edges[-1] - eps)
        return edges, centers, labels, transform

    if col in manual_hrm_bins:
        edges = manual_hrm_bins[col]
        centers = np.arange(1, 8, dtype=float)
        labels = [str(int(c)) for c in centers]
        def transform(s):
            s = s.astype(float)
            s = np.round(s)
            return np.clip(s, 1, 7)
        return edges, centers, labels, transform

    combined = pd.concat([full0[col], full1[col]], ignore_index=True).dropna()
    lo = np.floor(float(np.nanmin(combined)) * 10) / 10 if not combined.empty else 0.0
    hi = np.ceil(float(np.nanmax(combined)) * 10) / 10 if not combined.empty else 1.0
    bins = np.arange(lo, hi + 0.100001, 0.1, dtype=float)
    centers = bins[:-1] + 0.05
    labels = [f"{b:.1f}" for b in bins[:-1]]
    return bins, centers, labels, (lambda s: s)


top_feats = [c for c in sorted_cols if c in M][:2]
if len(top_feats) < 2:
    top_feats += [c for c in M if c not in top_feats][: (2 - len(top_feats))]

selected_cols = top_feats + out
selected_global_bins = {col: build_bins_for_column(col, DF_0, DF_1, out) for col in selected_cols}

ymax_by_col_small = {}
for col, (bins, _, _, transform) in selected_global_bins.items():
    maxc = 0
    for subset_mask in subsets.values():
        d0 = transform(DF_0.loc[subset_mask, col].dropna().astype(float))
        d1 = transform(DF_1.loc[subset_mask, col].dropna().astype(float))
        c0, _ = np.histogram(d0, bins)
        c1, _ = np.histogram(d1, bins)
        maxc = max(maxc, int(c0.max() if c0.size else 0), int(c1.max() if c1.size else 0))
    ymax_by_col_small[col] = maxc + 1

plot_style = {
    "font.size": 20,
    "axes.titlesize": 32,
    "axes.labelsize": 21,
    "xtick.labelsize": 20,
    "ytick.labelsize": 15,
    "legend.fontsize": 17,
    "figure.titlesize": 35,
}

selected_labels = {idx: f"{i+1}" for i, idx in enumerate(selected_idx)}

with plt.rc_context(plot_style):
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 20), sharex="col", sharey="col")
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.88, wspace=0.0, hspace=0.0)

    row_names = [name.replace(" Performance ", " Performance\n") for name in subsets.keys()]

    for row_idx, (subset_name, mask) in enumerate(subsets.items()):
        df0_sub = DF_0[mask]
        df1_sub = DF_1[mask]

        for col_idx, col in enumerate(selected_cols):
            ax = axes[row_idx, col_idx]

            data0 = df0_sub[col].dropna().astype(float)
            data1 = df1_sub[col].dropna().astype(float)

            bins, centers, labels, transform = selected_global_bins[col]
            data0 = transform(data0)
            data1 = transform(data1)

            counts0, _ = np.histogram(data0, bins)
            counts1, _ = np.histogram(data1, bins)

            bin_w = (bins[1] - bins[0]) if len(bins) > 1 else 1.0
            width = bin_w * 0.4

            ax.bar(
                centers - width / 2,
                counts0,
                width=width,
                color="salmon",
                label="Originals" if (row_idx == 0 and col_idx == 0) else "",
                align="center",
            )
            ax.bar(
                centers + width / 2,
                counts1,
                width=width,
                color="skyblue",
                label="MOCEs" if (row_idx == 0 and col_idx == 0) else "",
                align="center",
            )

            # Label selected employees on BOTH Original and MOCE bars.
            idx_order = [i for i in selected_idx if i in DF_0.index[mask]]

            for frame, counts, offset, label_color in (
                (DF_0, counts0, -width / 2, "#9a241d"),
                (DF_1, counts1, +width / 2, "#174c7e"),
            ):
                vals = transform(
                    frame.loc[mask, col].reindex(idx_order).dropna().astype(float)
                )

                stacked = {}

                for idx, value in vals.items():
                    bidx = int(np.digitize(value, bins) - 1)
                    bidx = max(0, min(bidx, len(counts) - 1))

                    k = stacked.get(bidx, 0)

                    ax.text(
                        centers[bidx] + offset,
                        counts[bidx] + 0.8 + k * 1.3,
                        selected_labels[idx],
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=18,
                        color=label_color,
                    )

                    stacked[bidx] = k + 1

            ax.set_xticks(centers)
            if row_idx == nrows - 1:
                ax.set_xticklabels(labels, rotation=45)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if col_idx == 0:
                ax.set_ylabel("# of Employees")
            else:
                ax.tick_params(axis="y", labelleft=False)

            ax.set_ylim(0, ymax_by_col_small[col] + 5)

            if row_idx == 0:
                ax.set_title(col)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left")

    for r in range(nrows):
        axes[r, 0].text(
            -0.25, 0.5, row_names[r],
            rotation=90, va="center", ha="center",
            transform=axes[r, 0].transAxes,
            fontweight="bold",
            clip_on=False,
        )

    fig.savefig(evaluation_path / f"{setting_slug}_figure.png", dpi=300, bbox_inches="tight")
    plt.show()