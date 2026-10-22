# Multi-Outcome Counterfactual Explanations for Decision Support in Human Resource Management

This repository contains the **data and Python source code** for the empirical analyses accompanying a research manuscript. It implements **multi-outcome counterfactual explanations (MOCEs)** using fitted predictive models and constrained optimization. The analyses compare MOCEs with single-outcome counterfactual explanations (CEs) and sequential single-outcome procedures, and examine sensitivity to alternative outcome requirements.

The three case studies illustrate distinct decision-support settings:

| Setting | Dataset | Variables allowed to change | Predicted outcomes | Predictive model |
| --- | --- | --- | --- | --- |
| Employee development planning | `Data-Skills.csv` (138 employees) | Seven skills; integer increases only, within 1–5 | Performance and job satisfaction | Multi-output Random Forest: 150 trees, depth 4, min. leaf 6 |
| Diagnostic explanation | `Data-Big5.csv` (268 employees) | Five Big Five personality traits; changes in either direction, within 1–7 | Performance and stress | Multi-output Random Forest: 150 trees, depth 4, min. leaf 6 |
| Organizational intervention design | `Data-HRM.csv` (264 employees) | Fifteen HRM practices; changes in either direction, within 1–7 | Performance and turnover intention | Multi-output Random Forest: 150 trees, depth 6, min. leaf 3 |

In each setting, other employee characteristics remain fixed during counterfactual optimization. **Big Five profiles are diagnostic contrasts, not proposed personality interventions.** The HRM and skills examples are model-based decision-support exercises, not causal estimates of intervention effects.

## Repository structure

```text
MOCE/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── Data-Skills.csv
├── Data-Big5.csv
├── Data-HRM.csv
├── Skills.py
├── Big Five Personality.py
├── HRM Practices.py
├── Cross-setting Summary.py
├── compare_predictive_models.py
├── selected_models.py
├── moce_evaluation.py
├── ce_helpers.py
├── constraint_learning.py
└── embed_mip.py
```

- **`Skills.py`, `Big Five Personality.py`, `HRM Practices.py`**: Fit the selected predictive model, validate predictive performance, compute single-outcome CEs, sequential CEs, and MOCEs, assess threshold sensitivity, and generate illustrative cases and distribution figures.
- **`selected_models.py`**: Stores the selected Random Forest configurations, fits the multi-output models to each full analytical dataset, and exports the tree constraints needed by the optimizer.
- **`compare_predictive_models.py`**: Compares six predictive model families (CART, Random Forest, Gradient Boosting, MLP, linear regression, and LinearSVR) with repeated cross-validation. This is a *model-selection analysis*, separate from the counterfactual optimization runs. It does **not** automatically rewrite `selected_models.py`.
- **`moce_evaluation.py`**: Implements common validation, CE/sequential/MOCE comparisons, threshold sensitivity, result reporting, and optional checkpointing.
- **`ce_helpers.py`, `constraint_learning.py`, `embed_mip.py`**: Implement the optimization formulation and embedding of fitted models into mathematical optimization problems.
- **`Cross-setting Summary.py`**: Combines output tables from all three case studies after they have been run.

## Installation

**Python 3.11** is recommended. The counterfactual optimization requires a working **Gurobi** installation and a valid license; the code uses Pyomo's `gurobi_persistent` solver interface.

From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements list the necessary Python packages without claiming that any particular set of package versions has been tested across platforms. Gurobi also requires its own working license configuration; installing `gurobipy` alone may not be sufficient.

## Running the analyses

Run the commands **from the repository root**, where the `Data-*.csv` files are located.

### 1. Optional: compare predictive models

```bash
python compare_predictive_models.py
```

This evaluates the six predictive model families using **three repetitions of five-fold cross-validation**. Candidate models are ranked by the mean of the outcome-specific mean R² values. The configurations used in the case-study experiments are explicitly recorded in `selected_models.py`; they are **not** updated automatically by this comparison script.

For a shorter test of the model-selection pipeline (not the full model-selection experiment):

```bash
python compare_predictive_models.py --quick
```

### 2. Run the three case studies

```bash
python "Skills.py"
python "Big Five Personality.py"
python "HRM Practices.py"
```

Each case-study script:

1. Fits its selected Random Forest using the full analytical sample and evaluates predictive adequacy using repeated cross-validation.
2. Evaluates outcome-specific CEs, both orders of sequential CEs, and MOCE under the **main** outcome requirements.
3. Runs separate MOCE analyses for **lenient** and **demanding** requirements.
4. Produces employee-level and aggregate CSV outputs, selected-case comparisons, and a four-by-four Original-versus-MOCE histogram figure.

The outcome requirements currently specified in the case-study scripts are:

| Setting | Lenient | Main | Demanding |
| --- | --- | --- | --- |
| Skills | Performance ≥ 3.70; satisfaction ≥ 4.20 | Performance ≥ 3.80; satisfaction ≥ 4.30 | Performance ≥ 3.90; satisfaction ≥ 4.40 |
| Big Five | Performance ≥ 6.00; stress ≤ 4.50 | Performance ≥ 6.10; stress ≤ 4.40 | Performance ≥ 6.20; stress ≤ 4.30 |
| HRM practices | Performance ≥ 5.90; turnover intention ≤ 3.10 | Performance ≥ 6.00; turnover intention ≤ 3.00 | Performance ≥ 6.10; turnover intention ≤ 2.90 |

The selected illustrative employee indices (zero-based dataset row positions) are **[36, 4, 42]** for Skills, **[25, 136, 234]** for Big Five, and **[43, 219, 86]** for HRM Practices. They are defined inside the individual case-study scripts and can be changed there.

**Runtime note:** Exact optimization of embedded Random Forests can be computationally expensive, particularly in the 15-variable HRM setting. The HRM script uses progress reporting and checkpoints so that completed employee-level calculations can be reused after interruption. Retain the generated checkpoints locally if you intend to resume a long run.

### 3. Combine the three settings

Only after all three case-study scripts have produced their evaluation outputs, run:

```bash
python "Cross-setting Summary.py"
```

This generates compact cross-setting CSVs. It does **not** rerun the predictive models or optimizations.

## Generated files

Outputs are written under `results/` **at runtime**. Generated models, figures, logs, checkpoint files, and evaluation CSVs are excluded from version control by `.gitignore`; **the repository distributes source code and data, not completed experiment results**.

Each case study generates the following files in `results/evaluation/`, with `<setting>` equal to `skills`, `big_five`, or `hrm_practices`:

| File | Description |
| --- | --- |
| `<setting>_predictive_adequacy_repeats.csv` | Repeated out-of-fold validation metrics |
| `<setting>_predictive_adequacy_summary.csv` | Predictive metrics compared with the outcome-mean baseline |
| `<setting>_ce_comparison_instances.csv` | Instance-level CE, sequential CE, and MOCE results |
| `<setting>_ce_comparison_summary.csv` | Aggregated method comparisons |
| `<setting>_selected_case_comparison.csv` | Method comparisons for illustrative employees |
| `<setting>_threshold_sensitivity.csv` | Lenient, main, and demanding threshold results |
| `<setting>_setting_formulation.csv` | Recorded formulation and model specifications |
| `<setting>_figure.png` | Four-by-four histograms comparing original and MOCE profiles |

The HRM script additionally writes local checkpoint files. The predictive-model selection script writes to `results/model_selection_six_models/`, while `selected_models.py` exports model artifacts under `results/`.

`Cross-setting Summary.py` combines the results into:

- `cross_setting_formulation.csv`
- `cross_setting_predictive_adequacy.csv`
- `cross_setting_ce_comparison.csv`
- `cross_setting_threshold_sensitivity.csv`

Some additional manuscript analyses (for example, choosing the better of the two sequential orders) require combining records in the instance-level comparison files rather than reading a single summary row.

## Methodological notes

- All selected Random Forest models are **multi-output** regressors with 150 trees and `random_state=42`.
- Predictive adequacy is evaluated with three repetitions of five-fold cross-validation against a training-fold outcome-mean baseline. The chosen model is subsequently refitted on the full analytical dataset for counterfactual analysis.
- MOCE enforces joint outcome requirements in the fitted predictive model, while sequential single-outcome procedures optimize the outcomes in a specified order; a later sequential step does not necessarily preserve earlier outcome validity.
- Change burden is measured using an equally weighted L1 distance across variables permitted to change. The feasible feature domains differ by setting, as described above.
- Big Five counterfactuals use a small directional adjustment (`0.0005`) when processing solutions to avoid numerical disagreement near fitted tree split thresholds; final predictions are checked against the requirements.
- Counterfactual validity refers to **predictions of the fitted model**, not observed employee outcomes, and should not be interpreted as causal or field-validated evidence.

## Attribution and license

Parts of the optimization helper code were adapted from the public [ce-ocl implementation](https://github.com/tabearoeber/ce-ocl). See the included [MIT License](LICENSE) for the repository's license.
