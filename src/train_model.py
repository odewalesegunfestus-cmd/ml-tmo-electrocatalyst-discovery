from pathlib import Path
import pandas as pd

from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor

# Locate project folder
project_root = Path(__file__).resolve().parent.parent

# Load dataset
csv_file = project_root / "data" / "demodata1.csv"
catalyst_data = pd.read_csv(csv_file)

# 9 input features
feature_columns = [
    "Ni_fraction",
    "Co_fraction",
    "Fe_fraction",
    "Calcination_temp_C",
    "Surface_area_m2g",
    "Oxygen_vacancy_percent",
    "Ni3_Ni2_ratio",
    "Co3_Co2_ratio",
    "ECSA_cm2"
]

X = catalyst_data[feature_columns]

# Target
y = catalyst_data["Overpotential_10mA_mV"]

# Train final GPR model on all 25 catalysts
gpr = GaussianProcessRegressor()
gpr.fit(X, y)

# Generate 5000 LHS virtual catalysts using Ni and Co
sampler = qmc.LatinHypercube(d=2, seed=42)
sample = sampler.random(n=10000)

virtual_data = []

for row in sample:
    Ni = row[0]
    Co = row[1]
    Fe = 1 - Ni - Co

    if Fe >= 0:
        virtual_data.append([Ni, Co, Fe])

    if len(virtual_data) == 5000:
        break

virtual_catalysts = pd.DataFrame(
    virtual_data,
    columns=["Ni_fraction", "Co_fraction", "Fe_fraction"]
)

# Add average values for the remaining 6 features
for column in feature_columns[3:]:
    virtual_catalysts[column] = catalyst_data[column].mean()

# Predict overpotential and uncertainty
prediction, uncertainty = gpr.predict(
    virtual_catalysts[feature_columns],
    return_std=True
)

virtual_catalysts["Predicted_Overpotential"] = prediction
virtual_catalysts["Uncertainty"] = uncertainty

# Rank catalysts from best to worst

ranked_catalysts = virtual_catalysts.sort_values(
    by="Predicted_Overpotential",
    ascending=True
)

print(ranked_catalysts.head(10))

# Create EI score

virtual_catalysts["EI_score"] = (
    virtual_catalysts["Predicted_Overpotential"]
    - virtual_catalysts["Uncertainty"]
)
# Rank by Expected Improvement

ei_ranked = virtual_catalysts.sort_values(
    by="EI_score",
    ascending=True
)

print("\nTOP 10 BY EXPECTED IMPROVEMENT\n")

print(ei_ranked.head(10))


# Create results folder if it does not exist
results_folder = project_root / "results"
results_folder.mkdir(exist_ok=True)

# Save Top 10 virtual catalysts
top10_virtual = ei_ranked.head(10)

top10_virtual.to_csv(
    results_folder / "top10_virtual_catalysts.csv",
    index=False
)

print("\nTop 10 virtual catalysts saved successfully.")

# Save all EI-ranked catalysts
ei_ranked.to_csv(results_folder / "ranked_virtual_catalysts.csv", index=False)

# Select best catalyst
best_catalyst = ei_ranked.iloc[0]

# Save best catalyst to text file
with open(results_folder / "best_virtual_catalyst.txt", "w") as file:
    file.write(str(best_catalyst))

print("\nBEST VIRTUAL CATALYST\n")
print(best_catalyst)

print("\nFiles saved successfully.")
