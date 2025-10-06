import json
import sys
import os
import pandas as pd
sys.path.append('modules')

from modules.data_analysis import business_insights, print_cluster_profiles, calculate_average_area, calculate_average_price

def extract_dashboard_metrics(dataframe):
    """Extract key metrics from the dataframe for dashboard display"""

    metrics = {}

    # Basic metrics
    metrics['total_records'] = len(dataframe)

    # Handle different column name possibilities for price
    price_columns = ['total', 'Total', 'aluguel', 'Aluguel']
    price_col = None
    for col in price_columns:
        if col in dataframe.columns:
            price_col = col
            break

    if price_col:
        # Clean price data if it contains currency symbols
        if dataframe[price_col].dtype == 'object':
            # Remove currency symbols and convert to numeric
            price_data = (dataframe[price_col]
                         .astype(str)
                         .str.replace("R$", "", regex=False)
                         .str.replace(".", "", regex=False)
                         .str.replace(",", ".", regex=False)
                         .str.strip())
            price_data = pd.to_numeric(price_data, errors='coerce')
        else:
            price_data = dataframe[price_col]

        metrics['average_price'] = calculate_average_price(dataframe)
    # else:
        # metrics['average_price'] = 0
        print("⚠️  No price column found in data")

    # Handle area column
    area_columns = ['area', 'Area', 'Área']
    area_col = None
    for col in area_columns:
        if col in dataframe.columns:
            area_col = col
            break

    # if area_col:
    metrics['average_area'] = calculate_average_area(dataframe)
    # else:
    #     metrics['average_area'] = 0
    #     print("⚠️  No area column found in data")

    metrics['total_clusters'] = dataframe['cluster'].nunique() if 'cluster' in dataframe.columns else 0

    # Cluster profiles
    if 'cluster' in dataframe.columns:
        cluster_profiles = []
        for cluster_id in sorted(dataframe['cluster'].unique()):
            cluster_data = dataframe[dataframe['cluster'] == cluster_id]

            cluster_profile = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'percentage': round(len(cluster_data)/len(dataframe)*100, 1),
                'avg_area': round(cluster_data['area'].mean(), 1),
                'median_area': round(cluster_data['area'].median(), 1),
                'avg_rent': round(cluster_data['total'].mean(), 0),
                'median_rent': round(cluster_data['total'].median(), 0),
                'avg_price_per_m2': round(cluster_data['cost_per_m2'].mean(), 0),
                'avg_rooms': round(cluster_data['quartos'].mean(), 1),
                'metro_access_pct': round(cluster_data['metro_proximo'].mean()*100, 0),
                'furnished_pct': round(cluster_data['mobiliado'].mean()*100, 0),
                'pet_friendly_pct': round(cluster_data['pet'].mean()*100, 0)
            }
            cluster_profiles.append(cluster_profile)

        metrics['cluster_profiles'] = cluster_profiles

        # Market segments (sorted by average rent)
        segments = sorted(cluster_profiles, key=lambda x: x['avg_rent'])
        segment_names = ['Economy', 'Mid-Market', 'Upper-Mid', 'Premium', 'Luxury']

        market_segments = []
        for i, segment in enumerate(segments):
            segment_name = segment_names[i] if i < len(segment_names) else f'Segment {i+1}'
            market_segment = {
                'name': segment_name,
                'cluster_id': segment['cluster_id'],
                'market_size': segment['size'],
                'market_percentage': segment['percentage'],
                'avg_rent': segment['avg_rent'],
                'avg_area': segment['avg_area'],
                'price_per_m2': segment['avg_price_per_m2'],
                'metro_access': segment['metro_access_pct']
            }
            market_segments.append(market_segment)

        metrics['market_segments'] = market_segments

    # Neighborhood analysis
    if 'bairro' in dataframe.columns and price_col:
        neighborhood_stats = []
        for neighborhood in dataframe['bairro'].unique():
            neighborhood_data = dataframe[dataframe['bairro'] == neighborhood]

            # Clean price data for this neighborhood
            if neighborhood_data[price_col].dtype == 'object':
                neighborhood_price_data = (neighborhood_data[price_col]
                                         .astype(str)
                                         .str.replace("R$", "", regex=False)
                                         .str.replace(".", "", regex=False)
                                         .str.replace(",", ".", regex=False)
                                         .str.strip())
                neighborhood_price_data = pd.to_numeric(neighborhood_price_data, errors='coerce')
            else:
                neighborhood_price_data = neighborhood_data[price_col]

            # Clean area data if needed
            if area_col:
                neighborhood_area_data = neighborhood_data[area_col]
                if neighborhood_area_data.dtype == 'object':
                    neighborhood_area_data = pd.to_numeric(neighborhood_area_data, errors='coerce')
            else:
                neighborhood_area_data = pd.Series([0] * len(neighborhood_data))

            # Calculate cost per m2 if both price and area are available
            cost_per_m2 = 0
            if area_col and neighborhood_area_data.mean() > 0:
                cost_per_m2 = neighborhood_price_data.mean() / neighborhood_area_data.mean()

            neighborhood_stat = {
                'name': neighborhood,
                'properties_count': len(neighborhood_data),
                'avg_rent': round(neighborhood_price_data.mean(), 0) if not pd.isna(neighborhood_price_data.mean()) else 0,
                'avg_area': round(neighborhood_area_data.mean(), 1) if not pd.isna(neighborhood_area_data.mean()) else 0,
                'avg_price_per_m2': round(cost_per_m2, 0)
            }
            neighborhood_stats.append(neighborhood_stat)

        # Sort by average rent
        neighborhood_stats.sort(key=lambda x: x['avg_rent'], reverse=True)
        metrics['neighborhood_stats'] = neighborhood_stats[:10]  # Top 10

    return metrics

def load_and_process_data():
    """Load and process the raw data to create clusters"""

    # Try to load already processed data with clusters first
    processed_files = [
        'processed_data_with_clusters.csv',
        'data_with_clusters.csv'
    ]

    for file_path in processed_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'cluster' in df.columns:
                    print(f"✅ Processed data with clusters loaded from: {file_path}")
                    return df
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue

    # If no processed data, check for raw data
    raw_files = [
        'tables/similar.houses.csv',
        'similar.houses.csv'
    ]

    df = None
    for file_path in raw_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Raw data loaded from: {file_path}")
                break
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue

    if df is None:
        print("❌ No data file found. Please ensure your data file is available.")
        print("Expected files:", raw_files + processed_files)
        return None

    # Check if clusters already exist
    if 'cluster' in df.columns:
        print("✅ Data already contains cluster information")
        return df

    # If no clusters, we need to create basic metrics without clustering
    print("⚠️  No cluster information found. Creating basic metrics only.")
    print("   To get full cluster analysis, run the main analysis first: python main.py")

    # Ensure we have the necessary columns for basic analysis
    required_columns = ['area', 'total']
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Missing required columns: {required_columns}")
        return None

    return df

def main():
    """Main function to generate dashboard data"""

    df = load_and_process_data()
    if df is None:
        return

    # Extract metrics
    metrics = extract_dashboard_metrics(df)

    # Save to JSON file
    output_file = 'dashboard_data.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Dashboard data saved to: {output_file}")

        # Print summary
        print(f"\n📊 Dashboard Metrics Summary:")
        print(f"   • Total Records: {metrics['total_records']:,}")
        print(f"   • Average Price: R$ {metrics['average_price']:,.0f}")
        print(f"   • Average Area: {metrics['average_area']:.1f} m²")
        print(f"   • Total Clusters: {metrics['total_clusters']}")

        if 'cluster_profiles' in metrics:
            print(f"   • Cluster Profiles: {len(metrics['cluster_profiles'])} clusters")

        if 'neighborhood_stats' in metrics:
            print(f"   • Neighborhood Stats: {len(metrics['neighborhood_stats'])} neighborhoods")

    except Exception as e:
        print(f"❌ Error saving dashboard data: {e}")

if __name__ == "__main__":
    main()