
import pandas as pd
import matplotlib.pyplot as plt

# 453 340 -----> parrot.png
# 532 532 -----> pitbull.jpeg
# 597 335 -----> Car.jpeg
# 540 720 -----> einstein.jpg
# 1920 1080 ---> Sunset.jpg

df = pd.read_csv("../app/ImgProcLabDemo/benchmark/results/gaussianV3.csv")

Size = [(453, 340), (532, 532), (597, 335),
        (540, 720), (1920, 1080)]

averages = []

for (Width, Height) in Size:
    filtered = df[
        (df["Width"] == Width) &
        (df["Height"] == Height) &
        (df["KernelSize"].between(3, 31))
    ]

    # Convert Version values into separate columns
    pivot = filtered.pivot(
        index="KernelSize",
        columns="Version",
        values="AverageMs"
    )

    # Plot execution time against kernel size
    pivot.plot(
        kind="line",
        marker="o",
        figsize=(10, 6)
    )

    plt.xlabel("Kernel Size")
    plt.ylabel("Average Execution Time (ms)")
    plt.title(f"Gaussian Filter — {Width}×{Height}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calculate average execution time for each implementation
    avg = pivot.mean()

    avg.name = f"{Width}×{Height}"
    averages.append(avg)

# Combine averages for all images
average_df = pd.DataFrame(averages)

# Display average execution times
print("\nAverage execution time per image (ms):")
print(average_df.to_string(float_format=lambda x: f"{x:.6f}"))

# Plot average execution time for each image
average_df.plot(
    kind="bar",
    figsize=(12, 6),
    rot=0
)

plt.xlabel("Image Size")
plt.ylabel("Average Execution Time (ms)")
plt.title("Gaussian Filter — Average Execution Time by Image Size")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Overall average execution time
print("\nOverall average execution time (ms):")
print(average_df.mean().to_string(float_format=lambda x: f"{x:.6f}"))
