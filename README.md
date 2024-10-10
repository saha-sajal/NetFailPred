
# Modeling and Prediction of Network Failures: A Machine Learning Approach

**Authors:**
- Chandrika Saha (Western University, Canada)
- Dr. Sajal Saha (University of Northern British Columbia, Canada)
- Dr. Anwar Haque (Western University, Canada)

## Overview

This project addresses the challenge of **network failure prediction** by creating a **synthetic network failure dataset** based on real-world patterns observed in network infrastructures. Network failures cause severe service disruptions, leading to financial losses, customer dissatisfaction, and business interruptions, especially during critical times. Our study simulates network traffic and generates synthetic failure data using Cisco's network failure guidelines. We applied several state-of-the-art **Machine Learning (ML)** and **Deep Learning (DL)** algorithms to predict failures, achieving high accuracy, precision, recall, and F1 scores.

The project provides:
1. **Synthetic network failure datasets** based on network topologies and real-world failure guidelines.
2. A **comprehensive ML/DL framework** for predicting network failures.
3. Source code for the simulation and prediction models.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Installation](#installation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Network failures can occur due to various factors such as hardware malfunctions, software errors, misconfigurations, or external causes like natural disasters. In this project, we simulate network traffic over a multi-year period, adhering to Cisco's network failure guidelines, to generate a **synthetic network failure dataset**. This dataset is used to train various **ML and DL models** to predict different types of failures and ensure the network remains operational with minimal downtime.

## Dataset

The **synthetic dataset** contains records of various failure types (CPU, memory, card, port, power supply, OS misconfiguration, etc.) based on traffic simulation for two different-sized network topologies. The dataset is available in CSV format and is split into training and test sets.

### Dataset Features:
- **Node utilization rates**: CPU, memory, card, port utilization, etc.
- **Risk factors**: Location, misconfiguration, OS upgrades, etc.
- **Failure types**: Power supply, memory, CPU, card, port, link, and more.

You can download the dataset [here](link-to-dataset).

## Methodology

The project consists of two main components:
1. **Network Traffic Simulation**: We simulate network traffic flow based on Cisco's failure guidelines for different network topologies (100 nodes and 200 nodes).
2. **ML/DL-based Prediction Framework**: We use the generated datasets to train various machine learning and deep learning models, including Logistic Regression, Naive Bayes, Gradient Boosting, QDA, and Deep Neural Networks (DNN).

## Models

### Machine Learning Models:
- Logistic Regression (LR)
- Naive Bayes (NB)
- Gradient Boosting (GB)
- Quadratic Discriminant Analysis (QDA)

### Deep Learning Models:
- **Deep Neural Networks (DNN)** with multiple hidden layers and dropout for regularization.

## Results

Our experiments show that **Deep Neural Networks (DNN)** outperform other models in predicting network failures, with accuracy and F1 scores ranging from 97% to 99%. These results demonstrate the potential of predictive techniques to reduce network downtime and optimize redundant network resources.

Performance comparison across models (Accuracy, Precision, Recall, F1):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| DNN   | 99%      | 98%       | 97%    | 97%      |
| LR    | 96%      | 95%       | 94%    | 94%      |
| GB    | 98%      | 98%       | 97%    | 98%      |
| NB    | 92%      | 93%       | 92%    | 92%      |

## How to Use

1. **Download the Dataset**: Use the provided link to download the synthetic dataset.
2. **Run the Models**: The Python scripts in this repository allow you to run ML and DL models on the dataset. Each model script includes:
   - Data preprocessing
   - Training and testing the model
   - Hyperparameter tuning
   - Evaluation metrics (accuracy, precision, recall, F1-score)

### Usage Instructions:
```bash
# Clone the repository
git clone https://github.com/username/network-failure-prediction.git
cd network-failure-prediction

```

## Installation

To set up the project locally:
1. Clone this repository.
2. Install the required dependencies.
3. Run the provided scripts to simulate the traffic, generate the dataset, and train the models.


