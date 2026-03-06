"""
Overture Data Downloader
========================

Downloads raw Overture Maps data (e.g. places, buildings, addresses) for a 
specified bounding box using the overturemaps Python library. 

Data is converted to a GeoDataFrame and can be exported as GeoJSON, Parquet, 
or CSV for downstream processing in the pipeline.

Usage:
    python src/download_overture.py --min_lon -122.34 --min_lat 47.62 --max_lon -122.32 --max_lat 47.64 
"""
import overturemaps
import geopandas as gpd
from pathlib import Path
import argparse
import time
DATA_DIR = Path("data/raw_overture")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_overture_data(bbox, output_file, theme="places", type="place"):
    """
    Downloads Overture Maps data for a given bounding box.
    """
    print(f"Downloading Overture '{theme}' (type='{type}') for bounding box: {bbox}")
    start_time = time.time()
    
    try:
        reader = overturemaps.record_batch_reader(type, bbox)
        table = reader.read_all()
        
        if table.num_rows == 0:
            print("No data found for this bounding box.")
            return None
            
        print(f"Downloaded {table.num_rows} records in {time.time() - start_time:.2f} seconds.")
        
        df = table.to_pandas()
        
        from shapely import wkb
        df['geometry'] = df['geometry'].apply(
            lambda x: wkb.loads(x) if isinstance(x, (bytes, bytearray)) else x
        )
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        if output_file.endswith('.geojson'):
            gdf.to_file(output_file, driver='GeoJSON')
        elif output_file.endswith('.parquet'):
            gdf.to_parquet(output_file)
        elif output_file.endswith('.csv'):
            gdf.to_csv(output_file, index=False)
        else:
            print(f"Unsupported extension for {output_file}. Saving as GeoJSON.")
            output_file = output_file + '.geojson'
            gdf.to_file(output_file, driver='GeoJSON')
            
        print(f"Data saved to {output_file}")
        return gdf
        
    except Exception as e:
        print(f"Error downloading Overture data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Overture Maps data for a bounding box.")
    parser.add_argument('--min_lon', type=float, required=True, help="Minimum Longitude (West)")
    parser.add_argument('--min_lat', type=float, required=True, help="Minimum Latitude (South)")
    parser.add_argument('--max_lon', type=float, required=True, help="Maximum Longitude (East)")
    parser.add_argument('--max_lat', type=float, required=True, help="Maximum Latitude (North)")
    parser.add_argument('--theme', type=str, default="places", help="Overture theme (default: places)")
    parser.add_argument('--type', type=str, default="place", help="Overture feature type e.g. place, building, address (default: place)")
    parser.add_argument('--output', type=str, default="data/raw_overture/overture_data.geojson", help="Output file path (supports .geojson, .parquet, .csv)")
    
    args = parser.parse_args()
    
    bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    
    download_overture_data(
        bbox=bbox, 
        output_file=args.output,
        theme=args.theme,
        type=args.type
    )

if __name__ == "__main__":
    main()
