
# Code for "Quantifying Fairness in Spatial Predictive Policing"

This repository contains the code used to develop the paper **"Quantifying Fairness in Spatial Predictive Policing"**. The algorithms used in this work include **SEPP**, **KDE**, and **Naive KDE**, implemented using code developed by the **QuantCrime Lab** at the University of Leeds. The code is available at: [QuantCrime GitHub](https://github.com/QuantCrimAtLeeds/PredictCode).

## Project Structure

This project has the following directory structure and stores the following information:

### 1. **DATOS**: Stores all raw, processed data, and results.
- **BOGOTA**: Contains raw and processed data from Bogota.
  - **NAIVE**: Contains training, test, and predicted **Naive KDE** models for Bogota data.
  - **KDE**: Contains training, test, and predicted **KDE** models for Bogota data.
  - **SEPP**: Contains training, test, and predicted **SEPP** models for Bogota data.
  - **RESULTADOS**: Stores all image results from Bogota data.

- **CHICAGO**: Contains raw and processed data from Chicago.
  - **NAIVE**: Contains training, test, and predicted **Naive KDE** models for Chicago data.
  - **KDE**: Contains training, test, and predicted **KDE** models for Chicago data.
  - **SEPP**: Contains training, test, and predicted **SEPP** models for Chicago data.
  - **RESULTADOS**: Stores all image results from Chicago data.

- **RESULTADOS**: Contains all image results from both Chicago and Bogota data.

### 2. **SCRIPTS**: Stores all scripts for processing the data in the **DATOS** folder.
- The **SCRIPTS** folder contains **Jupyter notebooks** (ipynb) segmented by **Bogota** or **Chicago**.
- Scripts include:
  - Data organization.
  - Data processing.
  - Prediction, training, and testing per model.
  - Fairness measurements.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/QuantifyingFairness.git
   ```

2. **Install dependencies**:
   Make sure you have Python 3.x installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

To replicate the experiments and results from the paper, follow these steps:

1. Navigate to the appropriate folder for the city you want to process (either **Bogota** or **Chicago**).
2. Run the Jupyter notebook that corresponds to your task (data processing, model training, or fairness measurements).

For example, to process the data for **Bogota**, you can run:
```bash
jupyter notebook scripts/organize_data_bogota.ipynb
```

Then, use the subsequent scripts for model training and evaluation.

## Results

The images and results for each city (Bogota and Chicago) are saved under the respective **RESULTADOS** folder. These include visualizations such as:

- Fairness evaluation results.
- Predicted crime maps.
- Model performance metrics.

### Example output

When running the code, you will see results similar to:
```plaintext
[INFO] Fairness metrics for Naive KDE model: [metric_values]
[INFO] Predicted crime density map saved to 'RESULTADOS/Bogota/naive_predicted.png'
```

## Contributing

If you wish to contribute to this repository or improve the code, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them.
4. Push your changes and submit a pull request with a description of your modifications.

## License

This code is licensed under the **MIT License**. See the `LICENSE` file for more details.

## References

- [QuantCrime Lab GitHub Repository](https://github.com/QuantCrimAtLeeds/PredictCode).
- **"Quantifying Fairness in Spatial Predictive Policing"** (Paper link or citation).

## Contact

If you have any questions about the code or the paper, feel free to contact:

- Email: your-email@domain.com
- GitHub: [your-username](https://github.com/your-username)
