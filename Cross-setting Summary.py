"""Combine evaluation outputs from the three empirical MOCE scripts.

Run Skills.py, Big Five Personality.py, and HRM Practices.py first. This script
then creates compact cross-setting CSVs that can be used as the basis for the
paper's formulation and empirical-summary tables.
"""

from pathlib import Path

import pandas as pd


EVALUATION_DIR = Path("results/evaluation")
SETTINGS = {
    "skills": "Skills",
    "big_five": "Big Five Personality",
    "hrm_practices": "HRM Practices",
}


def read_output(slug, suffix):
    path = EVALUATION_DIR / "{}_{}.csv".format(slug, suffix)
    if not path.exists():
        raise FileNotFoundError(
            "Missing {}. Run the corresponding empirical script first.".format(path)
        )
    frame = pd.read_csv(path)
    if "setting" not in frame.columns:
        frame.insert(0, "setting", SETTINGS[slug])
    return frame


def combine(suffix, output_name):
    combined = pd.concat(
        [read_output(slug, suffix) for slug in SETTINGS],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(EVALUATION_DIR / output_name, index=False)
    return combined


def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    formulation = combine(
        "setting_formulation",
        "cross_setting_formulation.csv",
    )
    predictive = combine(
        "predictive_adequacy_summary",
        "cross_setting_predictive_adequacy.csv",
    )
    comparison = combine(
        "ce_comparison_summary",
        "cross_setting_ce_comparison.csv",
    )
    thresholds = combine(
        "threshold_sensitivity",
        "cross_setting_threshold_sensitivity.csv",
    )

    print("\nCross-setting formulation:\n")
    print(
        formulation[
            [
                "setting",
                "decision_purpose",
                "outcomes",
                "main_acceptable_region",
                "interpretation",
            ]
        ].to_string(index=False)
    )

    print("\nCross-setting CE comparison:\n")
    print(
        comparison[
            [
                "setting",
                "method",
                "solution_available_rate",
                "focal_valid_rate",
                "joint_valid_rate",
                "mean_change_burden_l1",
                "mean_changed_variables",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nCreated:")
    for name in [
        "cross_setting_formulation.csv",
        "cross_setting_predictive_adequacy.csv",
        "cross_setting_ce_comparison.csv",
        "cross_setting_threshold_sensitivity.csv",
    ]:
        print("- {}".format(EVALUATION_DIR / name))


if __name__ == "__main__":
    main()
