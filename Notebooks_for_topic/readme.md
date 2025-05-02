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

The project directory structure is organized as follows to manage data, scripts, and results:

```
.
├── DATOS/
│   ├── BOGOTA/
│   │   ├── NAIVE/      # Trained models and Naive KDE predictions (Bogotá)
│   │   ├── KDE/        # Trained models and KDE predictions (Bogotá)
│   │   ├── SEPP/       # Trained models and SEPP predictions (Bogotá)
│   │   └── RESULTADOS/ # Image results for Bogotá
│   ├── CHICAGO/
│   │   ├── NAIVE/      # Trained models and Naive KDE predictions (Chicago)
│   │   ├── KDE/        # Trained models and KDE predictions (Chicago)
│   │   ├── SEPP/       # Trained models and SEPP predictions (Chicago)
│   │   └── RESULTADOS/ # Image results for Chicago
│   └── RESULTADOS/   # Consolidated image results (Bogotá and Chicago)
├── SCRIPTS/          # Processing and analysis scripts and notebooks
│   ├── Bogota/       # Notebooks specific to Bogotá data processing and analysis
│   └── Chicago/      # Notebooks specific to Chicago data processing and analysis
├── models/           # Python modules for model selection and initialization
├── fairness_measures/# Python modules for fairness metric calculation
├── optimization/     # Python modules for optimization algorithms (if applicable)
├── poisson/          # Python modules for Poisson probability functions (if applicable)
├── spatial_simulation/# Python modules for spatial simulation utilities (if applicable)
├── global_vars.py    # Global configuration variables
├── .gitignore        # File to ignore files and directories in Git
├── LICENSE           # License file
└── README.md         # This file

```

* **DATOS/**: Contains all necessary data to run the scripts, including raw, intermediate, and generated results (predictions, metrics, images).
* **SCRIPTS/**: Contains the Jupyter notebooks (`.ipynb`) that guide through the complete workflow, from data processing to fairness evaluation. They are organized by city.
* **Module Directories (.py):** Contain reusable functions and classes for model selection, fairness metric calculation, optimization, etc.

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

    (Make sure to generate a `requirements.txt` file listing all project dependencies, e.g., `pandas`, `numpy`, `geopandas`, `shapely`, `dateutil`, `sklearn`, `tqdm`, `open_cp`, `scipy`, `cvxpy`, `sparse`, `matplotlib`, `plotly`, `pickle`). You can generate it automatically if you already have the dependencies installed in an environment: `pip freeze > requirements.txt`).

## Running the Code

To replicate the experiments and results from the paper, you must run the Jupyter notebooks in the `SCRIPTS/` folder in the appropriate sequence. The structure is organized by city, so you will generally follow the notebooks within `SCRIPTS/Bogota/` or `SCRIPTS/Chicago/`.

1.  **Start Jupyter Notebook or JupyterLab:** Open your terminal in the project root (with the virtual environment activated) and run:

    ```bash
    jupyter notebook
    # or
    # jupyter lab
    ```

2.  **Navigate to the `SCRIPTS/` folder** in the Jupyter interface.

3.  **Run the notebooks in order** for the city of interest. For example, to process and analyze **Bogotá** data:

    * Open and run the notebooks in `SCRIPTS/Bogota/` in numerical order (if numbered) or logical order (data processing -> modeling -> fairness analysis).
    * For example:
        * `6. Data Real Bogota.ipynb` (Data Processing)
        * `X. Naive Model Bogota.ipynb` (Naive Modeling)
        * `Y. KDE Model Bogota.ipynb` (KDE Modeling)
        * `Z. SEPP Model Bogota.ipynb` (SEPP Modeling)
        * `W. Fairness Bogota.ipynb` (Fairness Analysis)

    Repeat the process for **Chicago** using the notebooks in `SCRIPTS/Chicago/`.

Ensure that the output directories specified in the scripts (within `DATOS/`) exist before running cells that save files.

## Results

The results generated by the notebooks, including metrics tables, visualizations, and saved predictions, are stored in the `RESULTADOS/` folders within each city (`DATOS/BOGOTA/RESULTADOS/`, `DATOS/CHICAGO/RESULTADOS/`) and potentially in a consolidated `DATOS/RESULTADOS/` folder.

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
