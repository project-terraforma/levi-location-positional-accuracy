import geopandas as gpd
import pandas as pd
from shapely import wkb
import os
import numpy as np

def load_data(filepath):
    """
    Loads places data from a Parquet file and processes it into a GeoDataFrame.
    Adapted from data_loader.py.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        
        # Convert WKB geometry to shapely objects if needed
        if 'geometry' in df.columns and df['geometry'].dtype == 'object':
            df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
            
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # Helper to safely extract from potential dictionary/list structures
        def safe_extract(val, key, default=None):
            if isinstance(val, dict):
                return val.get(key, default)
            return default

        # Process categories
        if 'categories' in gdf.columns:
            gdf['primary_category'] = gdf['categories'].apply(
                lambda x: x.get('primary', 'uncategorized') if isinstance(x, dict) else 'uncategorized'
            )
        
        # Process addresses
        if 'addresses' in gdf.columns:
            gdf['formatted_address'] = gdf['addresses'].apply(
                lambda x: x[0].get('freeform', '') if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) else ''
            )
            
        # Process names
        if 'names' in gdf.columns:
            gdf['place_name'] = gdf['names'].apply(lambda x: x.get('primary', '') if isinstance(x, dict) else str(x))

        return gdf

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def inspect_data(gdf):
    """
    Prints schema and basic information about the dataframe.
    Adapted from inspect_data.py.
    """
    print("\n--- Data Inspection ---")
    print("Schema Info:")
    print(gdf.info())
    print("\nFirst 5 rows:")
    print(gdf.head())
    print("\nColumns:")
    print(gdf.columns.tolist())
    print("-----------------------\n")

def process_and_export(gdf, output_base, sample_size=500):
    """
    Samples data, exports to GeoJSON and CSV.
    Adapted from prepare_sample.py and convert_to_csv.py.
    """
    # 1. Sampling
    if len(gdf) > sample_size:
        print(f"Sampling {sample_size} records from {len(gdf)} total...")
        sampled_gdf = gdf.sample(n=sample_size, random_state=42)
    else:
        print(f"Data size ({len(gdf)}) is less than sample size. Using all records.")
        sampled_gdf = gdf.copy()

    # 2. Select columns
    columns_to_keep = ['id', 'place_name', 'primary_category', 'formatted_address', 'geometry', 'confidence']
    existing_cols = [c for c in columns_to_keep if c in sampled_gdf.columns]
    export_gdf = sampled_gdf[existing_cols].copy()
    
    # Add original geometry WKT for reference
    export_gdf['original_geometry_wkt'] = export_gdf['geometry'].apply(lambda geom: geom.wkt)

    # Ensure output directory
    output_dir = os.path.dirname(output_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 3. Export GeoJSON
    geojson_path = f"{output_base}.geojson"
    print(f"Exporting GeoJSON to {geojson_path}...")
    export_gdf.to_file(geojson_path, driver='GeoJSON')

    # 4. Export CSV (for Google My Maps)
    csv_path = f"{output_base}.csv"
    print(f"Exporting CSV to {csv_path}...")
    
    # Prepare for CSV: extract lat/lon, drop geometry
    csv_df = pd.DataFrame(export_gdf.drop(columns='geometry'))
    csv_df['longitude'] = export_gdf.geometry.x
    csv_df['latitude'] = export_gdf.geometry.y
    
    # Rename for user friendliness (My Maps)
    rename_dict = {
        'place_name': 'Name',
        'primary_category': 'Category',
        'formatted_address': 'Address',
        'confidence': 'Confidence Score'
    }
    csv_df = csv_df.rename(columns={k: v for k, v in rename_dict.items() if k in csv_df.columns})
    
    csv_df.to_csv(csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "project_d_samples.parquet"
    OUTPUT_BASE = "data/ground_truth_pending" # Extension will be added
    
    # Path adjustment logic
    if not os.path.exists(INPUT_FILE) and os.path.exists(f"../{INPUT_FILE}"):
        INPUT_FILE = f"../{INPUT_FILE}"
        
    # Check if data directory needs adjustment
    if not os.path.exists(os.path.dirname(OUTPUT_BASE)) and os.path.exists(f"../{os.path.dirname(OUTPUT_BASE)}"):
         OUTPUT_BASE = f"../{OUTPUT_BASE}"

    # Execution
    if os.path.exists(INPUT_FILE):
        gdf = load_data(INPUT_FILE)
        inspect_data(gdf)
        process_and_export(gdf, OUTPUT_BASE)
    else:
        print(f"Error: Input file '{INPUT_FILE}' not found.")