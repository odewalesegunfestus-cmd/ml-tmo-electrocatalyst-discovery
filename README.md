# Machine Learning-Guided Discovery of Transition Metal Oxide Electrocatalysts

## Project Overview

This project demonstrates a beginner workflow for machine learning-assisted catalyst discovery. The goal is to predict oxygen evolution reaction (OER) catalyst performance using catalyst composition data and machine learning techniques.

## Research Motivation

The Oxygen Evolution Reaction (OER) is a major bottleneck in alkaline water electrolysis for green hydrogen production. Transition metal oxide catalysts containing Ni, Co, and Fe are promising low-cost alternatives to noble-metal catalysts. Machine learning can accelerate catalyst screening by identifying promising compositions before experimental testing.

## Dataset

This repository uses a small demonstration dataset containing Ni-Co-Fe catalyst compositions and their corresponding overpotential values.

Note: The dataset is intended for educational and workflow-development purposes and does not represent experimental research results.

## Methodology

Load catalyst dataset using Pandas.
Perform exploratory data analysis.
Train a Linear Regression machine learning model.
Predict catalyst overpotential.
Evaluate model performance using R² score.
Rank catalyst candidates according to predicted performance.
Export results and visualizations.

## Results

Current outputs include:

Catalyst performance predictions
Predicted vs Actual performance graph
Ranked catalyst candidates
Exported prediction tables
R² model evaluation score

## Repository Structure

data/ → Input catalyst dataset

src/ → Python source code

figures/ → Generated graphs and visualizations

results/ → Prediction and ranking outputs

README.md → Project documentation

requirements.txt → Required Python packages

## Future Work

Expand dataset using literature data
Implement train-test splitting
Add cross-validation
Apply Gaussian Process Regression (GPR)
Implement Bayesian Optimization
Perform virtual catalyst screening
Integrate experimental validation

## Author

Odewale Segun Festus

MSc Industrial Chemistry

Research Interests:
Machine Learning for Catalysis, Green Hydrogen Production, Transition Metal Oxides, Sustainable Energy Technologies