# Import Pandas for reading and working with catalyst dataset tables
import pandas as pd

# Import Matplotlib for creating and saving graphs
import matplotlib.pyplot as plt

# Import Linear Regression model for simple machine learning prediction
from sklearn.linear_model import LinearRegression

# Import R2 score for measuring model accuracy
from sklearn.metrics import r2_score


# Load the catalyst dataset from the data folder
data = pd.read_csv("data/catalyst_data.csv")


# Select the catalyst composition columns as input features
X = data[["Ni", "Co", "Fe"]]


# Select overpotential as the target value to be predicted
y = data["Overpotential_mV"]


# Create the Linear Regression model
model = LinearRegression()


# Train the model using catalyst composition and overpotential data
model.fit(X, y)


# Predict overpotential for all catalysts in the dataset
predictions = model.predict(X)


# Add the predicted values as a new column in the dataset
data["Predicted_Overpotential"] = predictions


# Calculate R2 score to evaluate model performance
score = r2_score(y, predictions)


# Print the R2 score in the terminal
print("R2 Score:")
print(score)


# Save the dataset with predictions into the results folder
data.to_csv("results/catalyst_predictions.csv", index=False)


# Create a clean graph size for publication-style output
plt.figure(figsize=(8, 6))


# Plot actual overpotential against predicted overpotential
plt.scatter(data["Overpotential_mV"], data["Predicted_Overpotential"])


# Add a diagonal reference line for perfect prediction
plt.plot([190, 250], [190, 250], "--")


# Label the x-axis
plt.xlabel("Actual Overpotential (mV)")


# Label the y-axis
plt.ylabel("Predicted Overpotential (mV)")


# Add graph title
plt.title("Linear Regression Model Performance")


# Add grid lines for easier reading
plt.grid(True)


# Save the graph as a high-resolution PNG file
plt.savefig("figures/predicted_vs_actual.png", dpi=300, bbox_inches="tight")


# Show the graph on screen
plt.show()


# Rank catalysts from lowest predicted overpotential to highest
ranked = data.sort_values(by="Predicted_Overpotential")


# Save the ranked catalyst table into the results folder
ranked.to_csv("results/ranked_catalysts.csv", index=False)


# Print a blank line for cleaner terminal output
print()


# Print the heading for the best catalyst candidates
print("TOP 5 CATALYSTS")


# Print the top five catalyst candidates
print(ranked[["Catalyst_ID", "Predicted_Overpotential"]].head())