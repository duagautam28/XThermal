import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("thermal_dataset.csv")

correlation=df.corr()
print(correlation)

plt.figure(figsize=(8,6))

plt.imshow(correlation, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(correlation.columns)),
           correlation.columns,
           rotation=45)

plt.yticks(range(len(correlation.columns)),
           correlation.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()