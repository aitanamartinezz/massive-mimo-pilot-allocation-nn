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

## Results

The proposed neural network–based pilot assignment approach was evaluated under
multiple system configurations and compared against three reference strategies:
optimal pilot assignment, random assignment, and a traditional k-beams–based method.

Performance was assessed using the **total system NMSE**, which reflects the overall
quality of channel estimation across all users.

### Baseline Configuration

For a representative configuration with:
- 10 user equipments (UEs)
- 3 pilot sequences
- Square cell of 50 meters per side

The following NMSE values were obtained:

| Pilot Assignment Strategy | NMSE |
|---------------------------|------|
| Random assignment         | 3.69 |
| k-beams                   | 2.25 |
| Neural network (proposed) | 1.99 |
| Optimal assignment        | 1.78 |

The neural network significantly outperforms the random baseline and the traditional
k-beams method, achieving performance close to the optimal exhaustive solution.

### Impact of System Parameters

Experimental results show that:

- **Increasing the number of UEs** leads to higher NMSE for all methods, highlighting
  the importance of intelligent pilot assignment in dense scenarios.
- **Increasing the number of pilot sequences** consistently reduces NMSE, as additional
  pilots provide more degrees of freedom to mitigate interference.
- **Varying the cell size** affects channel estimation quality due to changes in UE
  spatial distribution and interference patterns.

Across all evaluated scenarios, the neural network approach remains consistently close
to the optimal solution and clearly superior to random assignment.

These results indicate that the proposed model is capable of learning meaningful pilot
assignment strategies that generalize across different network configurations.

## Conclusions

This project demonstrates that neural networks can effectively approximate optimal
pilot assignment strategies in massive MIMO systems, achieving near-optimal NMSE
performance with significantly lower computational complexity.

Key conclusions include:

- The proposed neural network consistently outperforms random and traditional
  k-beams–based pilot assignment methods.
- The model is able to generalize across different numbers of users, pilot sequences,
  and cell sizes.
- Although exhaustive optimal assignment provides the lowest NMSE, its combinatorial
  complexity makes it impractical for large-scale systems.
- The neural network offers a scalable and computationally efficient alternative that
  achieves performance close to the optimal solution.

From a system-level perspective, improved pilot assignment leads to more accurate
channel estimation, reduced interference, and more efficient use of radio resources,
which are critical factors in next-generation wireless networks.

👩‍💻 Author:

Aitana Martínez
Bachelor’s Degree in Data Science and Engineering
