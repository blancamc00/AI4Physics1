import numpy as np
import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt

# Recreate the dataset
np.random.seed(42)
masses = np.random.uniform(1, 10, 100)
accelerations = np.random.uniform(-5, 5, 100)
forces = masses * accelerations + np.random.normal(0, 0.5, 100)  # Add some noise

data = pd.DataFrame({"mass": masses, "acceleration": accelerations, "force": forces})
X = data[["mass", "acceleration"]].values
y = data["force"].values

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[],
    loss="loss(x, y) = (x - y)^2",
    population_size=50,
    progress=True,
    model_selection="Pareto",  # Choose the best equation
    verbosity=1,
)

model.fit(X, y)

print(model.equations_)

eqs = model.equations_

# Sort by complexity
eqs = eqs.sort_values("complexity")

# Manual Pareto front: keep only models that improve on loss
pareto_eqs = []
best_loss = np.inf

for _, row in eqs.iterrows():
    if row["loss"] < best_loss:
        pareto_eqs.append(row)
        best_loss = row["loss"]

pareto_df = pd.DataFrame(pareto_eqs)

# Plot it!
plt.plot(pareto_df['complexity'], pareto_df['loss'], 'o-')
plt.xlabel("Complexity")
plt.ylabel("Loss")
plt.title("Pareto Front: Simplicity vs Accuracy")
plt.grid(True)
plt.show()
