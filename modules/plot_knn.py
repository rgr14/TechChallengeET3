import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.neighbors import NearestNeighbors 


def plot_knn_side_by_side(scaled_df, merged_df, user_scaled, kmeans, n_neighbors=10, save_path="plots/KNN.plot.png"):
    
    # Ensure user_scaled is DataFrame with correct cols
    feature_cols = scaled_df.drop(columns=["cluster"]).columns
    if not isinstance(user_scaled, pd.DataFrame):
        user_scaled = pd.DataFrame(user_scaled, columns=feature_cols)
    

    # PCA on full dataset
    pca_full = PCA(n_components=2, random_state=42)
    reduced_full = pca_full.fit_transform(scaled_df.drop(columns=["cluster"]))
    
    reduced_full_df = pd.DataFrame(reduced_full, columns=["PC1", "PC2"])
    reduced_full_df["cluster"] = scaled_df["cluster"].values
    
    user_reduced_full = pca_full.transform(user_scaled)
    

    # Get user cluster + neighbors
    user_cluster = int(kmeans.predict(user_scaled)[0])
    cluster_mask = scaled_df["cluster"] == user_cluster
    cluster_features = scaled_df.loc[cluster_mask].drop(columns=["cluster"])
    
    # KNN within the cluster (add +1 to remove self later)
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(cluster_features)
    distances, indices = knn.kneighbors(user_scaled)
    
    # Exclude the first neighbor (self)
    neighbor_positions = cluster_features.iloc[indices[0][1:]].index
    
    # PCA for cluster only
    pca_cluster = PCA(n_components=2, random_state=42)
    reduced_cluster = pca_cluster.fit_transform(cluster_features)
    
    reduced_cluster_df = pd.DataFrame(reduced_cluster, columns=["PC1", "PC2"], index=cluster_features.index)
    user_reduced_cluster = pca_cluster.transform(user_scaled)
    
    
    # Plotting side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT: Full dataset
    axes[0].scatter(
        reduced_full_df["PC1"], reduced_full_df["PC2"],
        c=reduced_full_df["cluster"], cmap="tab10", alpha=0.4, s=20, edgecolor="none"
    )
    axes[0].scatter(user_reduced_full[:, 0], user_reduced_full[:, 1],
                    c="black", marker="*", s=130, edgecolor="k", linewidth=1, zorder=10, label="User Input")
    axes[0].set_title("Full dataset (PCA-reduced)")
    axes[0].set_ylabel('Principal component 1')
    axes[0].set_xlabel('Principal component 2')
    axes[0].legend()
    
    # RIGHT: Only user's cluster
    axes[1].scatter(
        reduced_cluster_df["PC1"], reduced_cluster_df["PC2"],
        c="lightgray", s=12, alpha=0.4, label="Cluster Houses"
    )
    axes[1].scatter(
        reduced_cluster_df.loc[neighbor_positions, "PC1"],
        reduced_cluster_df.loc[neighbor_positions, "PC2"],
        c="blue", s=40, alpha=0.5, edgecolor="k", linewidth=0.5,
        label="Nearest Neighbors"
)
    axes[1].scatter(user_reduced_cluster[:, 0], user_reduced_cluster[:, 1],
                    c="black", marker="*", s=130, edgecolor="k", linewidth=1, zorder=10, label="User Input")
    axes[1].set_title(f"User's Cluster (#{user_cluster})")
    axes[1].set_xlabel('Principal component 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #plt.show()
    
    return merged_df.loc[neighbor_positions]
