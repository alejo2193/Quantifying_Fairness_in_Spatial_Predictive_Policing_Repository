# Quantifying Fairness in Spatial Predictive Policing

![Status Badge](https://img.shields.io/badge/Status-In%20Development-yellow) ![License Badge](https://img.shields.io/badge/License-MIT-blue) ![Version Badge](https://img.shields.io/badge/Version-1.0.0-informational)

This repository contains the code used to develop the research presented in the paper **"Quantifying Fairness in Spatial Predictive Policing"**. The project implements and evaluates spatial predictive models (SEPP, KDE, Naive KDE) on crime data from Bogotá and Chicago, focusing on quantifying and analyzing the fairness of predictions in different regions.

## Table of Contents

* [About the Project](#about-the-project)
* [Algorithms Used](#algorithms-used)
* [Project Structure](#project-structure)
* [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Installation Steps](#installation-steps)
* [Running the Code](#running-the-code)
* [Usage Examples](#usage-examples)
* [Results](#results)
* [License](#license)
* [References](#references)
* [Contact](#contact)

## About the Project

This project addresses the growing concern about fairness in spatial crime prediction systems. We implement and compare the performance and fairness of three predictive models: **SEPP**, **KDE** (Kernel Density Estimation), and **Naive KDE** (a simple baseline).

The research is conducted using real crime data from two distinct cities, Bogotá and Chicago, each with its own spatial and socioeconomic characteristics that influence the definition of protected/unprotected regions. The code allows replicating the data processing, model simulation with sliding time windows, and the application of fairness metrics to evaluate the results.

## Algorithms Used

The predictive algorithms implemented in this work are:

* **SEPP (Self-Exciting Point Process):** A model that considers both the history of past events and the influence of recent events to predict the probability of future events.
* **KDE (Kernel Density Estimation):** A non-parametric method for estimating the probability density function of a random variable, used here to estimate the spatial intensity of crimes.
* **Naive KDE:** A simplified version of KDE, often used as a baseline.

These models were implemented using base code developed by the **QuantCrime Lab** at the University of Leeds, available at: <https://github.com/QuantCrimAtLeeds/PredictCode>.

## Project Structure

The project directory structure is organized as follows to manage data, scripts, and custom libraries:

```
.
├── Experiment_Scripts/ # Experimental scripts with real data from Bogota and Chicago (Jupyter Notebooks)
├── Librerias/          # Custom libraries and functions
│   ├── fairness_measures/
│   ├── model/
│   ├── predictCode/    # Base code from University of Leeds
│   └── robust_predict/
├── Examples/           # Examples demonstrating model usage based on simulations
├── Data/               # Stores all raw, processed data, and results
│   ├── BOGOTA/
│   └── CHICAGO/
├── global_vars.py      # Global configuration variables
├── .gitignore          # File to ignore files and directories in Git
├── LICENSE             # License file
└── README.md           # This file

```

* **Experiment_Scripts:** This folder contains the experimental scripts using real data from Bogota and Chicago. These are primarily Jupyter notebooks (`.ipynb`) located directly within this folder, without subdirectories for cities.

* **Librerias:** This folder stores the code for custom libraries and functions developed for training models and calculating model fit metrics for fairness and performance. These are global libraries used for simulated data, as well as real data from Bogota and Chicago. It contains the following subdirectories:
    * `fairness_measures/`
    * `model/`
    * `predictCode/` (Base code developed by the University of Leeds)
    * `robust_predict/`

* **Examples:** This folder contains examples demonstrating how the implemented models work based on simulations. These examples are likely provided as Jupyter notebooks or scripts.

* **Data/**: (Renamed from DATOS) Contains all necessary data to run the scripts, including raw, intermediate, and generated results (predictions, metrics, images). It is further organized by city.

* **global_vars.py**: Defines global parameters and configurations used across various notebooks and scripts.

## Installation

To set up the project locally, follow these steps:

### Prerequisites

Ensure you have the following installed:

* **Python 3.7+** (Python 3.8 or higher is recommended)
* **pip** (Python package installer)
* **git** (version control system)
* An environment to run Jupyter notebooks (e.g., Jupyter Notebook, JupyterLab, VS Code with Python/Jupyter extension).

### Installation Steps

1.  **Clone the repository:** Open your terminal or command prompt and run:

    ```bash
    git clone [https://github.com/alejo2193/Quantifying_Fairness_in_Spatial_Predictive_Policing_Repository.git](https://github.com/alejo2193/Quantifying_Fairness_in_Spatial_Predictive_Policing_Repository.git)
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd Quantifying_Fairness_in_Spatial_Predictive_Policing_Repository
    ```

3.  **Create a virtual environment (recommended):** It's a good practice to use virtual environments to manage project dependencies.

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:

    ```bash
    # On macOS/Linux
    source venv/bin/activate
    ```bash
    # On Windows
    .\venv\Scripts\activate
    ```

4.  **Install dependencies:** With the virtual environment activated, install the necessary libraries.

    ```bash
    pip install -r requirements.txt
    ```

    (Make sure to generate a `requirements.txt` file listing all project dependencies, for example, `pandas`, `numpy`, `geopandas`, `shapely`, `dateutil`, `sklearn`, `tqdm`, `open_cp`, `scipy`, `cvxpy`, `sparse`, `matplotlib`, `plotly`, `pickle`). You can generate it automatically if you already have the dependencies installed in an environment: `pip freeze > requirements.txt`).

## Running the Code

To replicate the experiments and results from the paper, you must execute the Jupyter notebooks in the `Experiment_Scripts/` folder in the appropriate sequence.

1.  **Start Jupyter Notebook or JupyterLab:** Open your terminal in the project root (with the virtual environment activated) and run:

    ```bash
    jupyter notebook
    # or
    # jupyter lab
    ```

2.  **Navigate to the `Experiment_Scripts/` folder** in the Jupyter interface.

3.  **Execute the notebooks in order** as per the workflow described in the paper or the notebooks themselves.

Ensure that the output directories specified in the scripts (within `Data/`) exist before running cells that save files.

## Usage Examples

Examples demonstrating the use of the implemented models based on simulations are provided within the **Examples** folder. These examples serve as practical guides for understanding how the models function.

## Results

The results generated by the notebooks, including tables of metrics, visualizations, and saved predictions, are stored in the `Data/` folder structure, typically within `Data/BOGOTA/RESULTADOS/`, `Data/CHICAGO/RESULTADOS/`, and potentially a consolidated `Data/RESULTADOS/` folder.

These results allow examining:

* The performance of the predictive models.
* The spatial distribution of predictions.
* The calculated fairness metrics (Max-Min, Gini, EMD, etc.).
* Fairness comparisons between models and between cities.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file in the repository root for full details.

## References

* **Paper:** "Quantifying Fairness in Spatial Predictive Policing" (Include link or full citation to the paper when available).
* **open_cp base code:** QuantCrime Lab at the University of Leeds repository: [https://github.com/QuantCrimAtLeeds/PredictCode](https://github.com/QuantCrimAtLeeds/PredictCode).
* Other relevant works or datasets (if applicable).

## Contact

If you have questions, comments, or suggestions about this project or the paper, please contact:

* **Author:** Diego Alejandro Hernandez Castaneda
* **Email:** dieahernandezcas@unal.edu.co
* **GitHub:** [alejo2193](https://github.com/alejo2193) ([https://github.com/alejo2193](https://github.com/alejo2193))
