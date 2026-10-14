# Multi-Outcome Counterfactual Explanations for Decision Support in Human Resource Management

This repository provides the data and Python code for the empirical analyses accompanying a manuscript under anonymous review. It implements multi-outcome counterfactual explanations (MOCEs) using predictive models and constrained optimization. The analyses compare MOCEs with single-outcome and sequential counterfactual explanations and examine sensitivity to the specified outcome requirements.

The three case studies address different decision-support settings:

| Setting | Data | Counterfactual variables | Predicted outcomes | Predictive model |
| --- | --- | --- | --- | --- |
| Employee development planning | `Data-Skills.csv` (138 observations) | Seven skills | Performance and job satisfaction | Multi-output decision tree (CART) |
| Diagnostic explanation | `Data-Big5.csv` (268 observations) | Five personality traits | Performance and stress | Multi-output decision tree (CART) |
| Organizational intervention design | `Data-HRM.csv` (264 observations) | Fifteen HRM practices | Performance and turnover intention | Multi-output neural network (MLP) |

## Repository contents

```text
MOCE/
├── README.md
├── LICENSE
├── Data-Skills.csv
├── Data-Big5.csv
├── Data-HRM.csv
├── Skills.py
├── Big Five Personality.py
├── HRM Practices.py
├── Cross-setting Summary.py
├── moce_evaluation.py
├── ce_helpers.py
├── constraint_learning.py
├── embed_mip.py
└── run_MLmodels.py
```

The three case-study scripts train the predictive models, evaluate predictive adequacy, generate counterfactuals, and run the main-threshold and threshold-sensitivity analyses. `moce_evaluation.py` handles the shared evaluation procedures; the other helper modules support model training and embedding predictive models in optimization problems. `Cross-setting Summary.py` combines the case-study results into cross-setting CSV summaries.

## Requirements

- Python 3.11 is recommended.
- A working Gurobi installation with a valid license is required for the counterfactual optimization (accessed through Pyomo's `gurobi_persistent` interface).
- Install the Python packages used by the three case-study pipelines:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn pyomo gurobipy
```

## Reproducing the analyses

Download or clone this repository, open a terminal in its root directory, and run:

```bash
python "Skills.py"
python "Big Five Personality.py"
python "HRM Practices.py"
python "Cross-setting Summary.py"
```

Run the three case-study scripts before the cross-setting summary script. Each case study is self-contained and reads its corresponding `Data-*.csv` file. Model fitting and optimization may take time; runtimes depend on the computer, solver installation, and license.

### Generated files

The case-study scripts create files under `results/`, including model-related outputs and the following evaluation results in `results/evaluation/` (where `<setting>` is `skills`, `big_five`, or `hrm_practices`):

| Output | Contents |
| --- | --- |
| `<setting>_predictive_adequacy_repeats.csv` | Repeated cross-validation results |
| `<setting>_predictive_adequacy_summary.csv` | Predictive performance compared with a mean-prediction baseline |
| `<setting>_ce_comparison_instances.csv` | Instance-level single-outcome, sequential, and MOCE comparisons |
| `<setting>_ce_comparison_summary.csv` | Aggregated counterfactual comparison metrics |
| `<setting>_selected_case_comparison.csv` | Profiles for the illustrative cases |
| `<setting>_threshold_sensitivity.csv` | Results for lenient, main, and demanding outcome requirements |
| `<setting>_setting_formulation.csv` | Recorded decision-setting specifications |

After all three scripts finish, `Cross-setting Summary.py` creates:

- `cross_setting_ce_comparison.csv`
- `cross_setting_predictive_adequacy.csv`
- `cross_setting_threshold_sensitivity.csv`
- `cross_setting_formulation.csv`

The case-study scripts also display the observed-versus-MOCE distribution plots using Matplotlib. They do not automatically save those plots as image files. Some manuscript tables, including the best-of-two sequential benchmark, require aggregation of the instance-level output rather than direct use of a single cross-setting summary row.

## Evaluation notes

The scripts use fixed model specifications and random seeds, and evaluate predictive accuracy with three repetitions of five-fold cross-validation. Counterfactual burden is measured with an equal-weighted L1 distance. Under the main thresholds, the evaluation distinguishes outcome-specific counterfactuals, sequential procedures (including both outcome orders), and MOCEs. Alternative lenient and demanding thresholds are evaluated separately.

Counterfactual results describe predictions of the fitted models, not observed effects of interventions. The empirical settings illustrate the decision-support framework; the generated profiles should not be interpreted as independently validated causal recommendations. Runtime figures may vary across machines and solver configurations.

## Attribution and license

Parts of the optimization helper code were adapted from the publicly available [ce-ocl](https://github.com/tabearoeber/ce-ocl) implementation. This repository is provided under the terms of the included [MIT License](LICENSE).
