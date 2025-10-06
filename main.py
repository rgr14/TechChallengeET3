import os
import json
import warnings
import pandas as pd
from sklearn.cluster import KMeans
from modules.plot_knn import plot_knn_side_by_side
from modules.elbow_method import run_elbow
from modules.data_transformation import scale_data
from modules.compute_neighbors import find_similar_houses
from modules.plot_kmeans_clusters import plot_clusters
from modules import data_analysis

def main(user_input):
    # Suppress only FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    ### Load & Merge Data
    ## Define path to CSV files
    path = "./database/data/"

    # list all data in data base folder
    files = [f for f in os.listdir(path) if f.endswith(".csv")]

    # concatenate all data frames into a single one
    dfs = []
    for f in files:
        filepath = os.path.join(path, f)
        df = pd.read_csv(filepath)
        # add a column for neighborhood (bairro)
        neighborhood = os.path.splitext(f)[0].replace("quintoandar_", "")
        df["bairro"] = neighborhood.split('_')[0]
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    
    ### Data wrangling

    ## Rename columns for easier access
    merged_df = merged_df.rename(columns={
        "Aluguel": "aluguel",
        "Condomínio": "condominio",
        "IPTU": "iptu",
        "Seguro incêndio": "seguro_incendio",
        "Taxa de serviço": "taxa_servico",
        "Total": "total"
    })

    ## Preview
    print(merged_df.shape)             # Print the shape of the merged DataFrame (number of rows and columns)
    print(list(merged_df.columns))     # Print the list of column names

    ## Count NaNs per column
    print(merged_df.isna().sum())

    ## Sort by aluguel and total for easier browsing
    merged_df = merged_df.sort_values(by=["aluguel", "total"], ignore_index=True)
    print(merged_df.head()) 

    ## Reset index once so everything stays aligned
    merged_df = merged_df.reset_index(drop=True)
    print("✅ Data loaded and merged successfully")

    data_refined = merged_df.copy()

    ## Drop unnecessary columns
    data_refined = data_refined.drop(["url", 'status', 'title', 'address_street', 'vagas', 'andar'], axis=1)

    ## Check data types 
    print("✅ Data structure before cleaning:")
    print(data_refined.info())
    
    ## Replace "Incluso" with "0" before numeric conversion
    data_refined["condominio"] = (
        data_refined["condominio"]
        .replace("Incluso", "0")   
    )

    ## Remove currency symbol and transform to numeric for columns 8 to 13
    data_refined.iloc[:, list(range(7, 13))] = (
        data_refined.iloc[:, list(range(7, 13))]
        .apply(lambda col: col.str.replace("R$", "", regex=False).str.strip())
        .apply(lambda col: col.str.replace(".", "", regex=False).str.strip())
        .apply(lambda col: col.str.replace(",", ".", regex=False).str.strip())
    )

    ## Convert "sim"/"não" to 1/0 for columns 5 to 7
    data_refined.iloc[:, list(range(4, 7))] = data_refined.iloc[:, list(range(4, 7))].replace({"sim": 1, "não": 0})

    ## Reorder columns
    new_order = list(range(4, 7)) + list(range(0, 4)) + list(range(7, 14))
    data_refined = data_refined.iloc[:, new_order]

    print(data_refined.head()) 
    
    ## Convert to numeric
    # Converter apenas colunas que não são numéricas
    data_refined.iloc[:, :-1] = data_refined.iloc[:, :-1].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    data_refined["aluguel"] = pd.to_numeric(data_refined["aluguel"], errors="coerce")
    data_refined["area"]    = pd.to_numeric(data_refined["area"], errors="coerce")

    ## Calculate cost per m2 and round to 2 decimal places
    data_refined["cost_per_m2"] = (data_refined["aluguel"] / data_refined["area"]).round(2)

    ## Check for any NaN values in the DataFrame
    print("NaN values per column before dropping rows:")
    print(data_refined.isna().sum())

    ## Drop any row with at least one NaN
    data_refined = data_refined.dropna() 
    data_refined = data_refined.reset_index(drop=True)

    print("NaN values per column after dropping rows:")
    print(data_refined.isna().sum())

    print("Visualizatioon of cleaned data (top 5 rows):")
    print(data_refined.head())
    
    ### exploratory data analysis 
    print("="*20 + 'Exploratory data analysis', '='*20)

    data_analysis.ensure_plots_dir()
    
    # # plot total rent distribution
    data_analysis.plot_rent_distribution(data_refined)

    # # plot area distribution
    data_analysis.plot_area_distribution(data_refined)

    # # plot price per neighborhood
    data_analysis.plot_price_per_neighborhood(data_refined)

    # # plot correlation matrix
    data_refined = data_refined.drop(['bairro'], axis=1)
    data_analysis.plot_correlation_matrix(data_refined)

    ### Scaling 

    scaled_df, preprocessor = scale_data(data_refined)
    scaled_df = scaled_df.dropna() 
    print("✅ Data scaled and cleaned successfully")
    print(scaled_df.head())

    # ### Elbow Method

    k_elbow = run_elbow(scaled_df, isoutliers=True)
    print("✅ Elbow method completed, optimal k: ", k_elbow)

    ###  K-Means clustering

    kmeans = KMeans(n_clusters=k_elbow, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(scaled_df)
    scaled_df["cluster"] = cluster_labels
    print("✅ KMeans clustering completed")

    ### Run KNN to find similar houses (k = 25)

    similar_houses, user_scaled, user_cluster, neighbor_positions, neighbor_labels = find_similar_houses(
        user_input,      # user's requirements for accomodation (number of rooms, etc)
        merged_df,       # full dataset with urls & titles
        data_refined,    # numeric dataset
        preprocessor,    # fitted ColumnTransformer
        kmeans,          # fitted KMeans
        scaled_df,       # scaled numeric dataset
        n_neighbors=25   # number of similar houses to be suggested
    )
    
    ### Plot clusters

    plot_clusters(scaled_df, user_scaled, kmeans, user_cluster, isoutliers=True)
    print("✅ KMeans cluster plot saved to : plots/ directory")

    print("Checking KMeans clustering results:")
    print(   "Unique cluster labels in scaled_df:", scaled_df["cluster"].unique())
    print(   "Counts per cluster:")
    print("   ", scaled_df["cluster"].value_counts())


    ## Remove outliers and re-run KMeans
    ## Merge cluster labels back into original data (keeps alignment!)

    data_with_clusters = data_refined.copy()
    data_with_clusters["cluster"] = scaled_df["cluster"]

    ## Identify rare clusters (e.g., clusters with <= 2 samples)
    rare_clusters = data_with_clusters["cluster"].value_counts()
    rare_clusters = rare_clusters[rare_clusters <= 2].index
    print("Rare clusters (<=2 samples):", rare_clusters.tolist())

    print(data_with_clusters[data_with_clusters["cluster"].isin(rare_clusters)])

    ## Identify index of outlier rows

    outlier_idx = data_with_clusters[data_with_clusters["cluster"].isin(rare_clusters)].index

    print("Outlier indices:", outlier_idx.tolist())

    ## Remove outliers from both original and scaled dataframes
    scaled_df = scaled_df.drop(index=outlier_idx)
    data_with_clusters = data_with_clusters.drop(index=outlier_idx)
    data_refined = data_refined.drop(index=outlier_idx)

    ### Re-run KMeans on cleaned data
    ("✅  Outliers removed, re-running KMeans...")

    #### Scaling 

    scaled_df, preprocessor = scale_data(data_refined)
    scaled_df = scaled_df.dropna() 
    print("✅ Data scaled and cleaned successfully")
    print(scaled_df.head())

    ### Elbow Method

    k_elbow = run_elbow(scaled_df, isoutliers=False)
    print("✅ Elbow method completed, optimal k: ", k_elbow)

    ###  K-Means clustering

    kmeans = KMeans(n_clusters=k_elbow, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(scaled_df)
    scaled_df["cluster"] = cluster_labels
    #data_refined["cluster"] = cluster_labels
    print("✅ KMeans clustering completed")
    
 
    ### Run KNN to find similar houses (k = 25)
  
    similar_houses, user_scaled, user_cluster, neighbor_positions, neighbor_labels = find_similar_houses(
        user_input,
        merged_df,       # full dataset with ursls & titles
        data_refined,    # numeric dataset
        preprocessor,    # fitted ColumnTransformer
        kmeans,          # fitted KMeans
        scaled_df,       # scaled numeric dataset
        n_neighbors=25
    )
    
    ### Plot clusters

    plot_clusters(scaled_df, user_scaled, kmeans, user_cluster, isoutliers=False)
    print("✅ KMeans cluster plot saved to : plots/ directory")

    ### Check KMeans clustering results:

    neighbors_records = plot_knn_side_by_side(scaled_df, merged_df, user_scaled, kmeans, n_neighbors=25)
    print("✅ KNN side-by-side plot saved to : plots/")
    
    ## Plot cluster features
    data_refined["cluster"] = cluster_labels
    
    # Cluster profiles 
    data_analysis.print_cluster_profiles(data_refined)

    # Bussiness insights
    data_analysis.business_insights(data_refined)

    # Boxplot of price for each cluster
    data_analysis.plot_clusters_price_boxplot(data_refined)

    # boxplot of area for each cluster
    data_analysis.plot_clusters_area_boxplot(data_refined)

    # boxplot of price per meter squared for each cluster
    data_analysis.plot_clusters_price_per_sm_boxplot(data_refined)

    # distribution of number of rooms for each cluster
    data_analysis.plot_room_dist_per_cluster(data_refined)

    #data_analysis.print_cluster_profiles(data_with_clusters)

    ##data_analysis.business_insights(data_refined)

    # Recommendations
    data_analysis.recommendations()

    # Save processed data with clusters for dashboard
    try:
        data_refined.to_csv('processed_data_with_clusters.csv', index=False)
        print("✅ Processed data with clusters saved to: processed_data_with_clusters.csv")
    except Exception as e:
        print(f"⚠️  Error saving processed data: {e}")

    # Generate dashboard data JSON
    try:
        import json
        from generate_dashboard_data import extract_dashboard_metrics

        print("📊 Generating dashboard data...")
        dashboard_metrics = extract_dashboard_metrics(data_refined)

        with open('dashboard_data.json', 'w', encoding='utf-8') as f:
            json.dump(dashboard_metrics, f, indent=2, ensure_ascii=False)

        print("✅ Dashboard data generated: dashboard_data.json")
        print(f"   • Total Records: {dashboard_metrics['total_records']:,}")
        print(f"   • Average Price: R$ {dashboard_metrics['average_price']:,.0f}")
        print(f"   • Average Area: {dashboard_metrics['average_area']:.1f} m²")
        print(f"   • Total Clusters: {dashboard_metrics['total_clusters']}")

    except Exception as e:
        print(f"⚠️  Error generating dashboard data: {e}")
