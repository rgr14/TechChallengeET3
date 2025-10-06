import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def plot_clusters(scaled_df, user_scaled, kmeans, user_cluster, figsize=(12,7), save_path="plots/kmeans_cluster_plot.png", isoutliers=False):

    """
    Robust PCA plot with discrete colors per cluster, decision regions, centroids and user point.
    - scaled_df: DataFrame of features + 'cluster' column (index aligned with original)
    - user_scaled: 1-row DataFrame or 1D/2D ndarray with same feature order as scaled_df.drop('cluster')
    - kmeans: fitted KMeans (trained on the same scaled_df features)
    - user_cluster: int cluster index predicted for user input (used for legend)
    """


    # 1) features and PCA
    features = scaled_df.drop(columns=["cluster"])
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(features)
    reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"], index=scaled_df.index)
    reduced_df["cluster"] = scaled_df["cluster"].values

    n_clusters = kmeans.n_clusters

    # 2) centroids projected to PCA space
    centroids_2d = pca.transform(kmeans.cluster_centers_)

    # 3) prepare mesh for decision regions
    pad = 1.0
    x_min, x_max = reduced_df["PC1"].min() - pad, reduced_df["PC1"].max() + pad
    y_min, y_max = reduced_df["PC2"].min() - pad, reduced_df["PC2"].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # 4) predict cluster on mesh
    Z = kmeans.predict(pca.inverse_transform(mesh_points)).reshape(xx.shape)

    # 5) build discrete colormap
    base = cm.get_cmap("tab20")
    color_list = base(np.linspace(0, 1, n_clusters))
    cmap = colors.ListedColormap(color_list)
    boundaries = np.arange(-0.5, n_clusters + 0.5, 1.0)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    # 6) plotting
    plt.figure(figsize=figsize)

    plt.contourf(xx, yy, Z, levels=np.arange(n_clusters + 1) - 0.5,
                 cmap=cmap, norm=norm, alpha=0.25)

    plt.scatter(
        reduced_df["PC1"], reduced_df["PC2"],
        c=reduced_df["cluster"], cmap=cmap, norm=norm,
        s=20, alpha=0.8, edgecolor="none"
    )

    # 7) centroids
    plt.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c="white", marker="*", s=130, edgecolor="k", linewidth=1
    )

    # 8) user input
    if isinstance(user_scaled, np.ndarray):
        user_arr = np.atleast_2d(user_scaled)
        user_df = pd.DataFrame(user_arr, columns=features.columns)
    else:
        user_df = user_scaled.reindex(columns=features.columns)
    user_reduced = pca.transform(user_df)

    plt.scatter(
        user_reduced[:, 0], user_reduced[:, 1],
        c="black", marker="*", s=130, edgecolor="k", linewidth=1, zorder=10
    )

    # 9) legend
    cluster_handles = [
        Patch(color=color_list[i], label=f"Cluster {i}")
        for i in range(n_clusters)
    ]
    centroid_handle = Line2D([0], [0], marker="*", color="w",
                             markerfacecolor="white", markeredgecolor="black",
                             markersize=12, linestyle="None", label="Centroids")
    user_handle = Line2D([0], [0], marker="*", color="w",
                         markerfacecolor="black",
                         markersize=12, linestyle="None", label=f"User Input (cluster {user_cluster})")

    plt.legend(handles=cluster_handles + [centroid_handle, user_handle], bbox_to_anchor=(1.02, 1), loc="upper left")

    # tidy up
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("KMeans clusters (PCA-reduced) with centroids and user input")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    # save
    
    if isoutliers:
        save_path = save_path.replace(".png", "_with_outliers.png")
    else:
        save_path = save_path.replace(".png", "_without_outliers.png")
    
    save_dir = os.path.dirname(save_path)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    #plt.show()

    return user_reduced