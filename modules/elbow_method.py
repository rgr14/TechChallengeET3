import os
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans

def run_elbow(scaled_data, number_k=20, save_path="plots/elbow_plot.png", isoutliers=False):
    inertia = []
    K = range(1, number_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # Find the elbow
    kl = KneeLocator(K, inertia, curve="convex", direction="decreasing")

    # Ensure plots folder exists
    plot_dir = os.path.dirname(save_path)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertia, "bo-")
    plt.vlines(kl.elbow, plt.ylim()[0], plt.ylim()[1],
               linestyles="dashed", colors="red")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow Method with detected optimal k")
    
    # Force x-axis values to integers
    plt.xticks(range(1, number_k + 1))

    # Save plot
    if isoutliers:
        save_path = save_path.replace(".png", "_with_outliers.png")
    else:
        save_path = save_path.replace(".png", "_without_outliers.png")
        
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("✅ Elbow plot saved to:", save_path)
    
    return kl.elbow
