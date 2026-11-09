import pandas as pd
import matplotlib.pyplot as plt

# 453 340 -----> parrot.png
# 532 532 -----> pitbull.jpeg
# 597 335 -----> Car.jpeg
# 540 720 -----> einstein.jpg
# 1920 1080 ---> Sunset.jpg
df = pd.read_csv("../app/ImgProcLabDemo/benchmark/results/gaussianV1.csv")

Size =[(453, 340), (532, 532), (597, 335), (540, 720), (1920, 1080)]

for ( Width, Height) in Size :
    # Select Full HD results, kernel sizes 3 to 15
    filtered = df[
        (df["Width"] == Width) &
        (df["Height"] == Height) &
        (df["KernelSize"].between(3, 31))]

    # Convert Version values into separate columns
    pivot = filtered.pivot(
        index="KernelSize",
        columns="Version",
        values="AverageMs")

    # Plot all implementations
    pivot.plot(
        kind="line",
        marker="o",
        figsize=(10, 6)
    )

    plt.xlabel("Kernel Size")
    plt.ylabel("Average Execution Time (ms)")
    plt.title(f"Gaussian Filter — {Width}×{Height}")
    plt.grid(True)
    plt.show()
