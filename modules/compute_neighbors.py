import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def find_similar_houses(
    user_input,
    original_dataset,
    refined_dataset,
    scaler,
    kmeans,
    processed_dataset,
    n_neighbors=25,
    save_path="tables/similar.houses.csv",
    weighting='inverse'
):
    """
    Find the most similar houses to user_input within the same cluster,
    including distance, similarity weight, and ranking based on both distance and weight.
    The final CSV is sorted by distance (closest first).
    """

    # Ensure same column order as refined_dataset
    user_df = pd.DataFrame([user_input], columns=refined_dataset.columns)

    # Scale input
    user_scaled = pd.DataFrame(
        scaler.transform(user_df),
        columns=user_df.columns
    )

    # Predict cluster
    user_cluster = int(kmeans.predict(user_scaled)[0])

    # Filter dataset to same cluster
    cluster_indices = processed_dataset.index[processed_dataset["cluster"] == user_cluster]
    cluster_features = refined_dataset.loc[cluster_indices]

    # Reset index for KNN alignment
    cluster_features_reset = cluster_features.reset_index(drop=True)

    # Scale cluster features
    cluster_scaled = pd.DataFrame(
        scaler.transform(cluster_features_reset),
        columns=cluster_features_reset.columns
    )

    # KNN within cluster
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    knn.fit(cluster_scaled)
    distances, indices = knn.kneighbors(user_scaled)

    # Flatten arrays
    distances = distances.flatten()
    indices = indices.flatten()

    # Map back to original indices
    neighbor_labels = cluster_indices[indices]

    # Compute weights
    eps = 1e-8
    if np.any(distances == 0):
        weights = (distances == 0).astype(float)
    else:
        if weighting == 'inverse':
            weights = 1 / (distances + eps)
        elif weighting == 'gaussian':
            sigma = np.mean(distances)
            weights = np.exp(-(distances**2) / (2 * sigma**2))
        elif weighting == 'uniform':
            weights = np.ones_like(distances)
        else:
            raise ValueError("weighting must be 'inverse', 'gaussian', or 'uniform'")

    # Normalize weights
    weights = weights / (weights.sum() + eps)

    # Retrieve neighbors from original dataset (URLs, titles, etc.)
    neighbors_full = original_dataset.loc[neighbor_labels].copy()
    neighbors_full["distance"] = distances
    neighbors_full["weight"] = weights
    neighbors_full["user_cluster"] = user_cluster

    # Add ranking columns
    neighbors_full = neighbors_full.reset_index(drop=True)
    neighbors_full["rank_distance"] = neighbors_full["distance"].rank(method='first', ascending=True).astype(int)
    neighbors_full["rank_weight"]   = neighbors_full["weight"].rank(method='first', ascending=False).astype(int)

    # Sort CSV by distance (closest first) while keeping rank columns
    neighbors_full = neighbors_full.sort_values(by="distance", ascending=True).reset_index(drop=True)

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    neighbors_full.to_csv(save_path, index=False)

    print(f"✅ KNN search completed (k = {n_neighbors}) within cluster {user_cluster}")
    print("   Table saved to:", save_path)

    return (
        neighbors_full,
        user_scaled,
        user_cluster,
        indices,
        neighbor_labels
    )