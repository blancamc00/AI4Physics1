import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data for Kepler's Third Law: T^2 ∝ r^3
np.random.seed(42)

# Random orbital radii between 0.1 and 30 AU
radii = np.random.uniform(0.1, 30, 50)

# Orbital periods (T) from T = sqrt(r^3)
periods = np.sqrt(radii ** 3)

# Add some noise to simulate measurement error
noisy_periods = periods + np.random.normal(0, 0.2, size=50)

# Derived quantities
T_squared = noisy_periods ** 2
r_cubed = radii ** 3

# Create dataframe
data_kepler = pd.DataFrame({
    "radius": radii,
    "period": noisy_periods,
    "T_squared": T_squared,
    "r_cubed": r_cubed
})

from pysr import PySRRegressor

X = data_kepler[["r_cubed"]].values
y = data_kepler["T_squared"].values

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[],
    model_selection="best",
    progress=True,
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
