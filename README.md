Dear reader,

This repository contains the data and the code to reproduce the analyses, tables, and plots reported in the article "Multi-Outcome Counterfactual Explanations in Human Resources Management and Organizational Behavior Research".

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Gurobi License](#gurobi-license)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This project provides an end-to-end implementation of multi-outcome counterfactual explanations (MOCEs) across three application settings (Skills, Big Five Personality, and HRM Practices). Given a predictive model for multiple outcomes, the scripts generate minimal, actionable changes that move an employee from predicted “non-success” to predicted “success” according to predefined multi-outcome bounds.

## Installation

### Prerequisites

Ensure you have the following installed on your PC:

- Python (version 3.11 recommended)
- Gurobi (required for the optimization / MOCE steps)

### Clone the Repository

Download or clone all files in this repository to your PC.

### Install Python Dependencies

pip install numpy pandas matplotlib scikit-learn gurobipy

## Usage

Run the application scripts from the repository root:

python "Skills.py"
python "Big Five Personality.py"
python "HRM Practices.py"

Each script:

- loads the corresponding dataset,
- trains/loads predictive models,
- generates MOCE recourse using a Gurobi-backed optimizer,
- prints compact summaries (e.g., selected employee tables, subset-level patterns),
- and produces the plots used in the paper.

## Gurobi License

These scripts call Gurobi through gurobipy to solve optimization problems. To run the MOCE optimization components, you must have:

- Gurobi installed
- an active Gurobi license configured on your machine.

If Gurobi is installed but not licensed, the scripts will fail at the optimization stage. For licensing options (including academic licenses), please consult Gurobi’s official documentation.

## Acknowledgements

Parts of the helper code (e.g., optimization / embedding utilities) were adapted from scripts in the following repository, with modifications to fit the present paper’s pipelines and settings:

https://github.com/tabearoeber/ce-ocl

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
