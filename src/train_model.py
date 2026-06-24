import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/catalyst_data.csv")

X = data[["Ni", "Co", "Fe"]]
y = data["Overpotential_mV"]

model = LinearRegression()
model.fit(X, y)

virtual_catalysts = pd.DataFrame(
    [
        [0.5, 0.3, 0.2],
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1]
    ],
    columns=["Ni", "Co", "Fe"]
)

predictions = model.predict(virtual_catalysts)

virtual_catalysts["Predicted_Overpotential"] = predictions

print(virtual_catalysts)