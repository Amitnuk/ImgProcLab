import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../app/ImgProcLabDemo/benchmark/results/gaussianV2.csv")

# Select Full HD results, kernel sizes 3 to 15
filtered = df[
    (df["Width"] == 540) &
    (df["Height"] == 720) &
    (df["KernelSize"].between(3, 31))
]

# Convert Version values into separate columns
pivot = filtered.pivot(
    index="KernelSize",
    columns="Version",
    values="AverageMs"
)

# Plot all implementations
pivot.plot(
    kind="line",
    marker="o",
    figsize=(10, 6)
)

plt.xlabel("Kernel Size")
plt.ylabel("Average Execution Time (ms)")
plt.title("Gaussian Filter — 453×340")
plt.grid(True)
plt.show()
