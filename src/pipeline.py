"""
End-to-End Prediction Pipeline
==============================

This script orchestrates the complete workflow for correcting place locations
within a given bounding box. 

Workflow:
  1. Downloads raw place data from Overture Maps
  2. Conflates OSM and Overture geometry to find building and street contexts
  3. Runs the ML model to predict corrected coordinates
  4. Post-processes the predictions to align pins along building facades
  5. Outputs the results as a CSV and an interactive Folium map

Usage:
    python src/pipeline.py --min_lon -122.34 --min_lat 47.62 --max_lon -122.32 --max_lat 47.64
"""
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
import folium
import json
from shapely.geometry import Point

# Import pipeline steps
from utils.download_overture import download_overture_data
from utils.get_conflated_repositioning import fetch_bbox_context, fetch_overture_bbox, process_batch_locations
from models.predict_location import load_model, predict_from_features
from utils.align_places import align_predictions

DATA_DIR = Path("data/raw_overture")
OUTPUT_DIR = Path("data/pipeline_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_pipeline_map(df, output_path):
    """Generate a folium map comparing Original -> Predicted -> Aligned."""
    if df.empty:
        return
        
    center_lat = df['original_lat'].mean()
    center_lon = df['original_lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='CartoDB positron')
    
    fg_orig = folium.FeatureGroup(name="Original Overture", show=True)
    fg_pred = folium.FeatureGroup(name="ML Predicted", show=True)
    fg_align = folium.FeatureGroup(name="Perfectly Aligned", show=True)
    
    for _, row in df.iterrows():
        name = row.get('place_name', 'Unknown')
        address = row.get('formatted_address', 'No Address')
        orig_lat, orig_lon = row['original_lat'], row['original_lon']
        pred_lat, pred_lon = row.get('pred_lat'), row.get('pred_lon')
        align_lat, align_lon = row.get('aligned_lat'), row.get('aligned_lon')
        
        popup_html = f"<b>{name}</b><br>{address}"
        
        # Original (Red)
        folium.Marker(
            location=[orig_lat, orig_lon],
            icon=folium.Icon(color='red', icon='times'),
            popup=f"Original:<br>{popup_html}"
        ).add_to(fg_orig)
        
        # Draw ML line if predicted
        if pd.notna(pred_lat) and pd.notna(pred_lon):
            # Draw Alignment string if aligned
            if row.get('was_aligned') and pd.notna(align_lat):
                folium.Marker(
                    location=[align_lat, align_lon],
                    icon=folium.Icon(color='green', icon='check'),
                    popup=f"Aligned ML:<br>{popup_html}"
                ).add_to(fg_align)
            else:
                folium.Marker(
                    location=[pred_lat, pred_lon],
                    icon=folium.Icon(color='blue', icon='star'),
                    popup=f"Predicted ML:<br>{popup_html}"
                ).add_to(fg_pred)

    fg_orig.add_to(m)
    fg_pred.add_to(m)
    fg_align.add_to(m)
    
    folium.LayerControl().add_to(m)

    m.save(str(output_path))
    print(f"Map saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="End-to-end Pipeline: Download -> Conflate -> Predict -> Align")
    parser.add_argument('--min_lon', type=float, required=True, help="Minimum Longitude (West)")
    parser.add_argument('--min_lat', type=float, required=True, help="Minimum Latitude (South)")
    parser.add_argument('--max_lon', type=float, required=True, help="Maximum Longitude (East)")
    parser.add_argument('--max_lat', type=float, required=True, help="Maximum Latitude (North)")
    args = parser.parse_args()
    
    bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    
    # --- Step 1: Download ---
    raw_file = DATA_DIR / f"overture_{args.min_lon}_{args.min_lat}.geojson"
    print("\n--- STEP 1: DOWNLOADING OVERTURE DATA ---")
    gdf = download_overture_data(bbox, str(raw_file), theme="places", type="place")
    
    if gdf is None or gdf.empty:
        print("No raw data found. Exiting pipeline.")
        return
        
    print(f"Loaded {len(gdf)} places.")
    
    # Prepare for Step 2
    raw_dicts = []
    import json
    import ast
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        
        def safe_eval_dict(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str) and val.strip().startswith('{'):
                try:
                    return json.loads(val)
                except:
                    try:
                        return ast.literal_eval(val)
                    except:
                        pass
            return {}
            
        names_dict = safe_eval_dict(row.get('names'))
        cat_dict = safe_eval_dict(row.get('categories'))
        
        d = {
            'id': row.get('id', f'raw_{idx}'),
            'place_name': names_dict.get('primary', 'Unknown'),
            'primary_category': cat_dict.get('primary', 'Unknown'),
            'original_lat': geom.y,
            'original_lon': geom.x,
            'gt_latitude': geom.y,
            'gt_longitude': geom.x,
            'original_geometry_wkt': geom.wkt
        }
        
        # Extract address safely
        addr_info = row.get('addresses', [])
        if hasattr(addr_info, 'tolist'):
            addr_info = addr_info.tolist()
            
        if isinstance(addr_info, str) and addr_info.strip().startswith('['):
             try:
                 addr_info = json.loads(addr_info)
             except:
                 try:
                     addr_info = ast.literal_eval(addr_info)
                 except:
                     addr_info = []
                 
        if isinstance(addr_info, list) and len(addr_info) > 0:
             addr_info = addr_info[0]
        if isinstance(addr_info, dict):
            parts = []
            if addr_info.get('freeform'):
                parts.append(addr_info['freeform'])
            else:
                if addr_info.get('street'): parts.append(addr_info['street'])
                if addr_info.get('locality'): parts.append(addr_info['locality'])
            d['formatted_address'] = ", ".join(parts)
        else:
            d['formatted_address'] = "No Address Provided"
            
        raw_dicts.append(d)
        
    # --- Step 2: Conflation ---
    print("\n--- STEP 2: CONFLATING OSM FEATURES ---")
    
    import time
    start_time = time.time()
    
    
    # Add a small buffer to the bbox for context fetching 
    ctx_bbox = (args.min_lon - 0.001, args.min_lat - 0.001, args.max_lon + 0.001, args.max_lat + 0.001)
    
    osm_bldgs, osm_addrs, streets, parking_lots = fetch_bbox_context(ctx_bbox)
    ov_bldgs, ov_addrs = fetch_overture_bbox(ctx_bbox)
    
    print(f"  Got {len(osm_bldgs)} OSM buildings, {len(osm_addrs)} OSM addresses.")
    print(f"  Got {len(ov_bldgs)} OV buildings, {len(ov_addrs)} OV addresses.")
    print(f"  Got {len(streets) if streets is not None else 0} street segments.")
    
    conflated_results = process_batch_locations(
        raw_dicts, osm_bldgs, osm_addrs, streets, parking_lots, ov_bldgs, ov_addrs
    )
        
    conflated_df = pd.DataFrame(conflated_results)
    
    print(f"Successfully conflated {conflated_df.get('matched_building', pd.Series(False)).sum()} places to buildings in {time.time() - start_time:.1f}s.")
    
    # --- Step 3: Prediction ---
    print("\n--- STEP 3: ML PREDICTION ---")
    model, model_info = load_model()
    if model is None:
        print("Model not found. Cannot proceed.")
        return
        
    predicted_results = []
    for _, row in conflated_df.iterrows():
        if row.get('matched_building'):
            pred_row = predict_from_features(row, model, model_info)
            predicted_results.append(pred_row)
        else:
            pred_row = dict(row)
            pred_row['pred_lat'] = row['original_lat']
            pred_row['pred_lon'] = row['original_lon']
            predicted_results.append(pred_row)
            
    pred_df = pd.DataFrame(predicted_results)
    
    # --- Step 4: Alignment ---
    print("\n--- STEP 4: ALIGNING PREDICTIONS ---")
    aligned_df = align_predictions(pred_df)
    
    aligned_count = aligned_df.get('was_aligned', pd.Series(False)).sum()
    print(f"Perfectly aligned {aligned_count} places into unified orientations.")
    
    # --- Step 5: Output ---
    print("\n--- STEP 5: SAVING OUTPUTS ---")
    out_csv = OUTPUT_DIR / f"pipeline_{args.min_lon}_{args.min_lat}.csv"
    out_map = OUTPUT_DIR / f"pipeline_{args.min_lon}_{args.min_lat}.html"
    
    aligned_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    
    generate_pipeline_map(aligned_df, out_map)
    
    print("\n✅ PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
