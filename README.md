# Neural Network–Based Pilot Assignment for Massive MIMO Systems

This repository presents a machine learning–based approach to optimize pilot sequence
assignment in **massive MIMO multi-user systems**, with the goal of reducing channel
estimation error measured by **Normalized Mean Squared Error (NMSE)**.

The project is based on my Bachelor’s Thesis in Data Science and Engineering and addresses
a core scalability challenge in next-generation wireless networks (5G / Beyond 5G / 6G).

---

## Problem Overview

In massive MIMO systems, the number of available pilot sequences is often lower than
the number of connected user equipments (UEs). As a consequence, pilot reuse becomes
necessary, introducing **pilot contamination**, which degrades channel estimation
accuracy and overall system performance.

Efficient pilot assignment is therefore a critical resource allocation problem in
dense multi-user scenarios.

---

## Proposed Approach

This project proposes a **supervised learning approach** based on **Artificial Neural
Networks (ANNs)** to learn pilot assignment strategies from simulated wireless scenarios.

The complete pipeline includes:
- Synthetic scenario and dataset generation
- Scenario-based data augmentation
- Optimal and random pilot assignment strategies
- Neural network training using TensorFlow/Keras
- Hyperparameter tuning using Keras Tuner (Hyperband)
- Model evaluation using NMSE

The neural network–based solution is compared against:
- **Optimal pilot assignment**, computed via exhaustive search
- **Random pilot assignment**, used as a baseline reference

This setup allows a clear assessment of whether the model learns meaningful allocation
patterns beyond random behavior.

---

## Evaluation Metrics

Model performance is evaluated using:
- **Normalized Mean Squared Error (NMSE)** for channel estimation quality
- Classification accuracy for pilot assignment
- Comparative analysis against optimal and random baselines

Results show that the neural network approach consistently reduces NMSE while maintaining
lower computational complexity than exhaustive optimal methods.

---

## Project Structure

```text
src/
├── data/
│   └── data_generation/
│       ├── scripts/
│       │   └── generate_dataset.py        # Main dataset generation entry point
│       ├── experiments/                   # Experimental and exploratory scripts
│       └── *.py                           # Core data generation and evaluation modules
├── training/
│   └── train_and_tune_model.py            # Model training and hyperparameter tuning
├── models/
├── evaluation/

data/
├── samples/
│   ├── optimal/                           # Sample datasets with optimal assignment
│   ├── random/                            # Sample datasets with random assignment
│   └── README.md
└── raw/                                   # Large datasets (local only, not tracked)

results/                                   # Figures and evaluation outputs
```

##  Quick Start 

Follow these steps to reproduce the main workflow of the project.

### 1. Generate sample datasets
```bash
python src/data/data_generation/scripts/generate_dataset.py
```
### 2. Train and tune the neural network
```bash
python src/training/train_and_tune_model.py
```
### 3. Inspect results

NMSE metrics and figures are saved in the results/ directory

Sample datasets used for reproducibility are available in data/samples/

- Sample Datasets

Small sample datasets are included in data/samples/ for demonstration and
reproducibility purposes.

Two types of datasets are provided:

 - Optimal pilot assignment
 - Generated via exhaustive search to minimize NMSE.

Random pilot assignment
Used as a baseline reference.

Dataset filenames follow the convention:

 - users{N}_antennas{M}.csv
where:

 - N is the number of user equipments (UEs)

 - M is the number of base station antennas

## Notes on Scalability

Dataset generation time increases combinatorially with the number of users and
pilot sequences. Larger datasets were generated offline for training and evaluation
but are not included due to computational and size constraints.

This behavior reflects a real-world scalability challenge in large-scale resource
allocation problems for wireless communication systems.

## Tech Stack

- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy, SciPy, Pandas
- Matplotlib

## Why This Project Matters

This project demonstrates how machine learning techniques can be applied to optimize
resource allocation in wireless communication systems, bridging theoretical
telecommunications models with data-driven optimization approaches.

It is particularly relevant for:

- 5G / Beyond 5G / 6G networks

- AI-driven network optimization

- Large-scale multi-user and high-density systems

👩‍💻 Author:

Aitana Martínez
Bachelor’s Degree in Data Science and Engineering
