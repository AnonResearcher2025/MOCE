# ============================================================
# Big Five MOCE pipeline
# ============================================================
# Purpose:
#   End-to-end Big Five MOCE pipeline used in the paper.
# Steps:
#   1) Load Data-Big5.csv
#   2) Fit multi-outcome predictive model
#   3) For employees outside the predicted success bounds, compute MOCE recourse
#   4) Construct DF_0 (baseline), DF_1 (recourse), DF_diff (DF_1 − DF_0) dataframes
#   5) Print (i) selected employees (baseline vs recourse), (ii) subset-level change patterns
#   6) Plot a compact 4×4 histogram grid (top-2 changed skills + outcomes), labeling 3 example employees
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

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# User settings
# ============================================================
t = 4
o = 2
rn = 42

# Indices (row numbers) of 3 example employees to annotate in the plot.
selected_idx = [98, 68, 11]

# MOCE settings
num_counterfactuals = 1
model = "cart"
mu = 0
obj = "l1"

# Actionable inputs (allowed to change)
M = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Imagination"]

# Only increase inputs
L = []

# Success bounds
lb = [6, 1]
ub = [7, 4.5]


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
data = pd.read_csv("Data-Big5.csv")
out = data.columns[-o:].tolist()
n = len(data.columns) - o
X = data.iloc[:, :n]
y = data[out]


# ============================================================
# Train predictive models
# ============================================================
y_pred = np.zeros(shape=(len(y), o))
d = {"target": "Outcomes", "numerical": data.columns[:n].tolist()}
I = [val for val in d["numerical"] if val not in M]

version = "V1"
alg = model
alg_list = [model]
algorithms = {"Outcomes": alg}

outcome_dict = {
    "Outcomes": {
        "task": "continuous",
        "X features": X.columns,
        "class": d["target"],
        "alg_list": alg_list,
        "X_train": X, "X_test": X,
        "y_train": y, "y_test": y,
    }
}

_silence = io.StringIO()
with contextlib.redirect_stdout(_silence):
    ch.train_models(outcome_dict, version)
    _ = ch.perf_trained_models(version, outcome_dict)
    y_pred, models = ch.load_model(algorithms, outcome_dict, None)

y_pred_df = pd.DataFrame(y_pred["Outcomes"], columns=out)


# ============================================================
# Model registry for the optimizer
# ============================================================
model_master = pd.DataFrame(columns=["outcome", "model_type", "save_path", "task", "objective"])
for i_out in range(len(out)):
    model_master.loc[len(model_master)] = [
        out[i_out],
        alg,
        f"results/{alg}/{version}_{out[i_out]}_model.csv",
        "continuous",
        0,
    ]
model_master["lb"] = lb
model_master["ub"] = ub
model_master["SCM_counterfactuals"] = None
model_master["features"] = [[col for col in X.columns]] * len(model_master)


# ============================================================
# Generate MOCEs
# ============================================================
condition = (
    (y_pred_df[out[0]] < lb[0]) | (y_pred_df[out[0]] > ub[0]) |
    (y_pred_df[out[1]] < lb[1]) | (y_pred_df[out[1]] > ub[1])
)
inds = np.where(condition.values)[0]

DF_pairs = []
_silence2 = io.StringIO()

def solve_moce_for_index(ind: int):
    with contextlib.redirect_stdout(_silence2):
        return ch.opt(
            X, X, X.loc[ind],
            d["numerical"],
            [],
            [],
            {},
            I,
            L,
            [],
            X.columns,
            True, mu, False, False,
            num_counterfactuals,
            model_master, None,
            obj=obj,
        )

for ind in inds:
    CEs, CEs_, _final_model = solve_moce_for_index(ind)

    row0 = pd.to_numeric(CEs_.iloc[0, :-o], errors="coerce")
    row1 = pd.to_numeric(CEs_.iloc[1, :-o], errors="coerce")
    diff = row1 - row0
    same = diff.abs() < 1e-15

    row1_adj = row1.copy()
    row1_adj.loc[~same & (diff > 0)] = row1.loc[~same & (diff > 0)] + 0.05
    row1_adj.loc[~same & (diff < 0)] = row1.loc[~same & (diff < 0)] - 0.05
    CEs_.iloc[1, :-o] = row1_adj.round(2)

    CEs_.iloc[0, -o:] = models["Outcomes"].predict(CEs_.iloc[[0], :-o].astype(float)).ravel()
    CEs_.iloc[1, -o:] = models["Outcomes"].predict(CEs_.iloc[[1], :-o].astype(float)).ravel()

    df = CEs_.copy()
    df.iloc[1:] = df.iloc[1:].where(df.iloc[1:] != df.iloc[0], "-")
    DF_pairs.append(df)

satisfy_inds = np.where(~condition.values)[0]
for ind in satisfy_inds:
    row = X.loc[ind].copy()
    for j in range(o):
        row[out[j]] = y_pred_df.iloc[ind, j]
    dash_row = pd.Series("-", index=row.index)
    DF_pairs.append(pd.DataFrame([row, dash_row]))

DF = pd.concat(DF_pairs, ignore_index=True)

DF_0 = DF.iloc[::2].reset_index(drop=True)
DF_1_raw = DF.iloc[1::2].reset_index(drop=True)
DF_1 = DF_1_raw.mask(DF_1_raw == "-", DF_0)
DF_diff = DF_1 - DF_0

DF_0 = DF_0.apply(pd.to_numeric, errors="coerce")
DF_1 = DF_1.apply(pd.to_numeric, errors="coerce")
DF_diff = DF_diff.apply(pd.to_numeric, errors="coerce")


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
    "High Performance Low Stress": subset1_mask,
    "Low Performance Low Stress": subset2_mask,
    "High Performance High Stress": subset3_mask,
    "Low Performance High Stress": subset4_mask,
}


# ============================================================
# Prints
# ============================================================
print_selected_employees_compact(DF_0, DF_1, selected_idx, out)
subset_summary = summarize_subset_patterns_solo_then_combos(DF_diff, subsets, M)
print_subset_patterns(subset_summary, subsets)


# ============================================================
# Plotting the figure
# ============================================================
manual_outcome_bins = {
    out[0]: np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], dtype=float),
    out[1]: np.array([1.0, 1.5, 2.5, 3.5, 4.501, 5.501], dtype=float),
}
manual_big5_bins = {
    "Extraversion":      np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float),
    "Agreeableness":     np.array([2.1, 2.9, 3.7, 4.5, 5.3, 6.1, 7.0], dtype=float),
    "Conscientiousness": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float),
    "Neuroticism":       np.array([1.0, 1.7, 2.7, 3.7, 4.7, 5.7, 6.004], dtype=float),
    "Imagination":       np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float),
}

def build_bins_for_column(col: str, full0: pd.DataFrame, full1: pd.DataFrame, out_cols: list):
    combined = pd.concat([full0[col], full1[col]], ignore_index=True).dropna()

    if col in manual_outcome_bins:
        edges = manual_outcome_bins[col]
        centers = (edges[:-1] + edges[1:]) / 2.0
        labels = [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]
        return edges, centers, labels, (lambda s: s)

    if col in manual_big5_bins:
        edges = manual_big5_bins[col]
        centers = (edges[:-1] + edges[1:]) / 2.0
        labels = [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]
        return edges, centers, labels, (lambda s: s)

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

            idx_order = [i for i in selected_idx if i in DF_0.index[mask]]
            vals0 = DF_0.loc[mask, col].reindex(idx_order).dropna().astype(float)
            vals1 = DF_1.loc[mask, col].reindex(idx_order).dropna().astype(float)
            vals0 = transform(vals0)
            vals1 = transform(vals1)

            stack0, stack1 = {}, {}
            for idx, v in vals0.items():
                lab = selected_labels.get(idx, str(idx))
                bidx = int(np.digitize(v, bins) - 1)
                bidx = max(0, min(bidx, len(counts0) - 1))
                x = centers[bidx] - (width / 2)
                k = stack0.get(bidx, 0)
                y = counts0[bidx] + 0.8 + k * 6
                ax.text(x, y, lab, ha="center", va="bottom", fontweight="bold")
                stack0[bidx] = k + 1

            for idx, v in vals1.items():
                lab = selected_labels.get(idx, str(idx))
                bidx = int(np.digitize(v, bins) - 1)
                bidx = max(0, min(bidx, len(counts1) - 1))
                x = centers[bidx] + (width / 2)
                k = stack1.get(bidx, 0)
                y = counts1[bidx] + 0.8 + k * 6
                ax.text(x, y, lab, ha="center", va="bottom", fontweight="bold")
                stack1[bidx] = k + 1

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
                ax.legend(loc="upper right")

    for r in range(nrows):
        axes[r, 0].text(
            -0.25, 0.5, row_names[r],
            rotation=90, va="center", ha="center",
            transform=axes[r, 0].transAxes,
            fontweight="bold",
            clip_on=False,
        )

    plt.show()