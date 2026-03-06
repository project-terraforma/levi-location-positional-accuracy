"""
Predict Corrected Location
===========================

This script takes a place (name, category, address, lat, lon) and predicts its corrected location.

Workflow:
  1. Downloads OSM data to compute repositioning strategy anchor points.
  2. Constructs the feature vector used during model training.
  3. Loads the trained direct regressor model to predict the optimal coordinate shift.
  4. Outputs the newly corrected latitude and longitude, applying building footprint snapping.

Usage:
    # Run prediction for a default test location
    python src/models/predict_location.py

    # Run prediction for a custom location
    python src/models/predict_location.py \
        --name "Starbucks" \
        --category "coffee_shop" \
        --address "123 Main St" \
        --lat 47.6062 \
        --lon -122.3321

    # Run predictions in batch mode from a CSV file
    python src/models/predict_location.py --csv data/new_locations.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import warnings
import json
import argparse
import sys

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

STRATEGY_NAMES = [
    'centroid', 'nearest_edge',
    'street_1_facing', 'street_2_facing', 'street_3_facing',
    'parking_facing',
    'south_edge', 'north_edge', 'east_edge', 'west_edge'
]


# ============================================================
# Utility Functions (same as train_model.py)
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def clamp_prediction(pred_lat, pred_lon, row, available_strategies, buffer_deg=0.0005):
    """Clamp predicted lat/lon within anchor bounding box + fallback to nearest_edge."""
    # Large-building guard: if the building is >20,000 sqm (e.g. shopping mall)
    # and the original point is already inside, don't move it at all.
    building_area = float(row.get('building_area_sqm', 0) or 0)
    building_wkt = row.get('building_wkt')
    if building_area > 20000 and pd.notna(building_wkt):
        try:
            from shapely import wkt
            from shapely.geometry import Point
            orig_pt = Point(row['original_lon'], row['original_lat'])
            poly = wkt.loads(building_wkt)
            if poly.contains(orig_pt):
                return row['original_lat'], row['original_lon']
        except Exception:
            pass

    anchor_lats, anchor_lons = [], []
    for s in available_strategies:
        lat_c, lon_c = f'{s}_lat', f'{s}_lon'
        if pd.notna(row.get(lat_c)) and pd.notna(row.get(lon_c)):
            anchor_lats.append(row[lat_c])
            anchor_lons.append(row[lon_c])

    if not anchor_lats:
        return pred_lat, pred_lon

    min_lat = min(anchor_lats) - buffer_deg
    max_lat = max(anchor_lats) + buffer_deg
    min_lon = min(anchor_lons) - buffer_deg
    max_lon = max(anchor_lons) + buffer_deg

    clamped_lat = np.clip(pred_lat, min_lat, max_lat)
    clamped_lon = np.clip(pred_lon, min_lon, max_lon)

    # For non-outdoor locations, snap to the building polygon directly
    category_type = row.get('category_type', 'other')
    is_outdoor = (category_type == 'outdoor')
    building_wkt = row.get('building_wkt')
    
    if not is_outdoor and pd.notna(building_wkt):
        try:
            from shapely import wkt
            from shapely.geometry import Point
            from shapely.ops import nearest_points
            
            poly = wkt.loads(building_wkt)
            pred_pt = Point(clamped_lon, clamped_lat)
            
            # nearest_points returns the nearest point on the polygon to pred_pt.
            # If pred_pt is already inside the polygon, it remains unchanged.
            pt_on_poly, _ = nearest_points(poly, pred_pt)
            clamped_lon, clamped_lat = pt_on_poly.x, pt_on_poly.y
            
            # Nudge points on the boundary inward perpendicular to the edge
            snapped_pt = Point(clamped_lon, clamped_lat)
            if poly.boundary.distance(snapped_pt) < 1e-8:
                boundary = poly.boundary
                proj_dist = boundary.project(snapped_pt)
                # Sample two nearby points on the boundary to get the tangent
                eps = 0.01
                p1 = boundary.interpolate(max(0, proj_dist - eps))
                p2 = boundary.interpolate(min(boundary.length, proj_dist + eps))
                tx, ty = p2.x - p1.x, p2.y - p1.y
                # Two candidate normals — pick the one pointing inward
                centroid = poly.centroid
                dx_c = centroid.x - clamped_lon
                dy_c = centroid.y - clamped_lat
                # Normal candidates: (ty, -tx) and (-ty, tx)
                if ty * dx_c + (-tx) * dy_c > 0:
                    nx, ny = ty, -tx
                else:
                    nx, ny = -ty, tx
                length = (nx**2 + ny**2)**0.5
                if length > 0:
                    nx, ny = nx / length, ny / length
                    # Move inward by 25% of building depth along this normal
                    depth = abs(dx_c * nx + dy_c * ny)
                    clamped_lon += 0.25 * depth * nx
                    clamped_lat += 0.25 * depth * ny

        except Exception:
            pass

    return clamped_lat, clamped_lon


# ============================================================
# Step 1: Get OSM Features for a Location
# ============================================================
def get_osm_features(lat, lon, place_name, category, address, confidence=0.9):
    """
    Run the OSM repositioning pipeline on a single location.
    Returns a dict of features (same columns as osm_repositioning_features.csv).
    """
    # Import the processing function from get_osm_repositioning.py
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.get_osm_repositioning import process_single_location

    # Build a row dict matching what process_single_location expects
    row = {
        'id': f'pred_{hash(f"{lat}_{lon}_{place_name}") % 100000:05d}',
        'place_name': place_name,
        'primary_category': category,
        'formatted_address': address,
        'confidence': confidence,
        'gt_latitude': lat,       # process_single_location uses gt_ coords for OSM download
        'gt_longitude': lon,
        'original_lat': lat,
        'original_lon': lon,
        'original_geometry_wkt': f'POINT ({lon} {lat})',
    }

    print(f"  Downloading OSM data for ({lat:.6f}, {lon:.6f})...")
    result = process_single_location(pd.Series(row))

    if not result.get('matched_building', False):
        error = result.get('error', 'Unknown error')
        print(f"  ⚠️  No building match: {error}")
        return None

    print(f"  ✓ Building matched ({result.get('building_match_method', '?')}, "
          f"{result.get('building_distance_m', 0):.1f}m)")
    print(f"  ✓ {result.get('num_strategies', 0)} strategy anchor points computed")

    return result


# ============================================================
# Step 2: Build Feature Vector (same as train_model.py)
# ============================================================
def build_feature_vector(osm_result, model_info):
    """
    Build the exact same feature vector used during training.
    Returns a dict with all features or None if impossible.
    """
    row = osm_result
    features = {}

    # ---- Context features ----
    conf = row.get('confidence')
    features['confidence'] = float(conf) if conf is not None else 0.5
    features['building_area_sqm'] = float(row.get('building_area_sqm') or 0.0)
    features['log_building_area'] = np.log1p(features['building_area_sqm'])
    features['building_distance_m'] = float(row.get('building_distance_m') or 50.0)
    features['building_match_address'] = int(row.get('building_match_method') == 'address_match')

    features['num_streets'] = int(row.get('num_streets') or 0)
    features['has_parking'] = int(bool(row.get('has_parking')))
    features['num_strategies'] = int(row.get('num_strategies') or 6)

    # Street features
    for i in range(1, 4):
        features[f'street_{i}_addr_match'] = int(
            row.get(f'street_{i}_match_method') == 'address_match'
        )
        features[f'street_{i}_distance_m'] = float(row.get(f'street_{i}_distance_m', 200))

    features['any_street_addr_match'] = int(
        features['street_1_addr_match'] or
        features['street_2_addr_match'] or
        features['street_3_addr_match']
    )

    # Address features
    address = row.get('formatted_address', '')
    if pd.isna(address):
        address = ''
    features['has_address'] = int(len(address) > 0)
    features['address_length'] = len(address)
    features['address_has_number'] = int(any(c.isdigit() for c in address))

    # Category encoding
    category_type_map = {
        'hotel': 'lodging', 'resort': 'lodging', 'inn': 'lodging',
        'bed_and_breakfast': 'lodging', 'lodge': 'lodging',
        'accommodation': 'lodging', 'cabin': 'lodging',
        'holiday_rental_home': 'lodging',
        'restaurant': 'food', 'fast_food_restaurant': 'food',
        'pizza_restaurant': 'food', 'mexican_restaurant': 'food',
        'chinese_restaurant': 'food', 'japanese_restaurant': 'food',
        'italian_restaurant': 'food', 'sandwich_shop': 'food',
        'bakery': 'food', 'cafe': 'food', 'coffee_shop': 'food',
        'burger_restaurant': 'food', 'bagel_shop': 'food',
        'sushi_restaurant': 'food', 'thai_restaurant': 'food',
        'ice_cream_shop': 'food', 'bar': 'food', 'pub': 'food',
        'convenience_store': 'retail', 'supermarket': 'retail',
        'grocery_store': 'retail', 'pharmacy': 'retail',
        'clothing_store': 'retail', 'hardware_store': 'retail',
        'furniture_store': 'retail', 'shoe_store': 'retail',
        'boutique': 'retail', 'department_store': 'retail',
        'shopping_center': 'retail', 'discount_store': 'retail',
        'professional_services': 'services', 'lawyer': 'services',
        'construction_services': 'services', 'marketing_agency': 'services',
        'campground': 'outdoor', 'rv_park': 'outdoor', 'park': 'outdoor',
        'farmers_market': 'outdoor', 'plaza': 'outdoor', 'dog_park': 'outdoor',
        'national_park': 'outdoor',
        'self_storage_facility': 'storage', 'storage_facility': 'storage',
    }
    category = row.get('primary_category', '')
    cat_type = category_type_map.get(category, 'other')
    row['category_type'] = cat_type

    # One-hot encoding — must match training columns exactly
    # Get the category columns from model_info
    all_feature_cols = model_info['feature_cols']
    cat_cols = [c for c in all_feature_cols if c.startswith('cat_')]
    for col in cat_cols:
        features[col] = int(col == f'cat_{cat_type}')

    original_lat = float(row.get('original_lat', row.get('gt_latitude')))
    original_lon = float(row.get('original_lon', row.get('gt_longitude')))

    available_strategies = model_info.get('available_strategies', STRATEGY_NAMES)
    for s in available_strategies:
        lat_c = f'{s}_lat'
        lon_c = f'{s}_lon'
        slat = row.get(lat_c)
        slon = row.get(lon_c)
        if pd.notna(slat) and pd.notna(slon):
            features[f'{s}_dlat'] = float(slat) - original_lat
            features[f'{s}_dlon'] = float(slon) - original_lon
        else:
            features[f'{s}_dlat'] = 0.0
            features[f'{s}_dlon'] = 0.0

    from itertools import combinations
    top_strategies = available_strategies[:5]
    for s1, s2 in combinations(top_strategies, 2):
        col = f'spread_{s1}_{s2}'
        features[col] = np.sqrt(
            (features.get(f'{s1}_dlat', 0) - features.get(f'{s2}_dlat', 0)) ** 2 +
            (features.get(f'{s1}_dlon', 0) - features.get(f'{s2}_dlon', 0)) ** 2
        )

    return features


# ============================================================
# Step 3: Load Model and Predict
# ============================================================
def load_model():
    """Load the trained direct regressor model."""
    import joblib

    model_path = MODEL_DIR / "direct_regressor.joblib"
    if not model_path.exists():
        print(f"     Run 'python src/train_model.py' first!")
        return None, None

    data = joblib.load(model_path)
    print(f"  ✓ Model loaded from {model_path}")
    print(f"    CV mean: {data.get('cv_mean_dist', '?'):.2f}m | "
          f"CV median: {data.get('cv_median_dist', '?'):.2f}m")
    return data['model'], data


def predict_single(osm_result, model, model_info):
    """
    Predict the corrected lat/lon for a single location.
    Returns (pred_lat, pred_lon, metadata_dict).
    """
    feature_cols = model_info['feature_cols']
    available_strategies = model_info.get('available_strategies', STRATEGY_NAMES)

    # Build feature vector
    features = build_feature_vector(osm_result, model_info)

    # Create feature array in correct column order
    X = np.array([[features.get(col, 0.0) for col in feature_cols]])

    # Predict Δlat, Δlon
    delta = model.predict(X)[0]
    d_lat, d_lon = delta[0], delta[1]

    original_lat = float(osm_result.get('original_lat', osm_result.get('gt_latitude')))
    original_lon = float(osm_result.get('original_lon', osm_result.get('gt_longitude')))

    raw_pred_lat = original_lat + d_lat
    raw_pred_lon = original_lon + d_lon

    # Clamp to anchor bounding box
    pred_lat, pred_lon = clamp_prediction(
        raw_pred_lat, raw_pred_lon, osm_result, available_strategies
    )
    # Collect metadata
    meta = {
        'original_lat': original_lat,
        'original_lon': original_lon,
        'predicted_lat': pred_lat,
        'predicted_lon': pred_lon,
        'raw_pred_lat': raw_pred_lat,
        'raw_pred_lon': raw_pred_lon,
        'delta_lat': d_lat,
        'delta_lon': d_lon,
        'was_clamped': (pred_lat != raw_pred_lat) or (pred_lon != raw_pred_lon),

        'building_match_method': osm_result.get('building_match_method', 'unknown'),
        'building_distance_m': osm_result.get('building_distance_m', 0),
        'num_strategies': osm_result.get('num_strategies', 0),
    }

    # Strategy anchor distances from predicted point
    for s in available_strategies:
        lat_c = f'{s}_lat'
        lon_c = f'{s}_lon'
        if pd.notna(osm_result.get(lat_c)) and pd.notna(osm_result.get(lon_c)):
            d = haversine_distance(pred_lat, pred_lon,
                                   float(osm_result[lat_c]),
                                   float(osm_result[lon_c]))
            meta[f'dist_to_{s}'] = d

    return pred_lat, pred_lon, meta


# ============================================================
# Main: CLI Interface
# ============================================================
def predict_and_display(name, category, address, lat, lon, confidence=0.9,
                        gt_lat=None, gt_lon=None):
    """Full prediction pipeline for a single location."""
    print("\n" + "=" * 70)
    print(f"  PREDICTING: {name}")
    print("=" * 70)
    print(f"  Category: {category}")
    print(f"  Address:  {address}")
    print(f"  Input:    ({lat:.7f}, {lon:.7f})")
    if gt_lat is not None:
        print(f"  GT:       ({gt_lat:.7f}, {gt_lon:.7f})")

    # 1. Load model
    print(f"\n  Loading model...")
    model, model_info = load_model()
    if model is None:
        return None

    # 2. Get OSM features
    print(f"\n  Fetching OSM features...")
    osm_result = get_osm_features(lat, lon, name, category, address, confidence)
    if osm_result is None:
        return {
            'name': name, 'original_lat': lat, 'original_lon': lon,
            'predicted_lat': lat, 'predicted_lon': lon,
            'status': 'no_building_match'
        }

    # 3. Predict
    print(f"\n  Running ML prediction...")
    pred_lat, pred_lon, meta = predict_single(osm_result, model, model_info)

    move_dist = haversine_distance(lat, lon, pred_lat, pred_lon)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  RESULT                                     │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Original:   ({lat:.7f}, {lon:.7f})  │")
    print(f"  │  Predicted:  ({pred_lat:.7f}, {pred_lon:.7f})  │")
    print(f"  │  Moved:      {move_dist:.1f}m                           │")
    if meta.get('was_clamped'):
        print(f"  │  ⚠ Prediction was clamped to building bbox  │")

    if gt_lat is not None:
        orig_err = haversine_distance(lat, lon, gt_lat, gt_lon)
        pred_err = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
        improvement = orig_err - pred_err
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │  Original error:  {orig_err:.1f}m                       │")
        print(f"  │  ML error:        {pred_err:.1f}m                       │")
        sign = "↓" if improvement > 0 else "↑"
        word = "improved" if improvement > 0 else "worse"
        print(f"  │  {sign} {abs(improvement):.1f}m {word:30s}│")

    print(f"  └─────────────────────────────────────────────┘")

    result = {
        'name': name,
        'category': category,
        'address': address,
        'original_lat': lat,
        'original_lon': lon,
        'predicted_lat': pred_lat,
        'predicted_lon': pred_lon,
        'move_distance_m': move_dist,
        'building_match': osm_result.get('building_match_method'),
        'num_strategies': osm_result.get('num_strategies'),
        'status': 'success',
    }
    if gt_lat is not None:
        result['gt_lat'] = gt_lat
        result['gt_lon'] = gt_lon
        result['original_error_m'] = orig_err
        result['predicted_error_m'] = pred_err
        result['improvement_m'] = improvement

    return result

def predict_from_features(row, model, model_info):
    """
    Predict the corrected lat/lon for a single location given an existing feature row.
    Returns the updated dictionary with predicted coordinates.
    """
    row_dict = dict(row)
    
    # 3. Predict
    pred_lat, pred_lon, meta = predict_single(row_dict, model, model_info)
    
    # Return everything
    row_dict['pred_lat'] = pred_lat
    row_dict['pred_lon'] = pred_lon
    
    return row_dict



def predict_batch(csv_path):
    """Predict for all locations in a CSV file."""
    print(f"\n  Loading locations from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  ✓ {len(df)} locations loaded")

    # Auto-detect column names
    lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat', 'original_lat']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon', 'original_lon']), None)
    name_col = next((c for c in df.columns if c.lower() in ['name', 'place_name']), None)
    cat_col = next((c for c in df.columns if c.lower() in ['category', 'primary_category']), None)
    addr_col = next((c for c in df.columns if c.lower() in ['address', 'formatted_address']), None)
    gt_lat_col = next((c for c in df.columns if c.lower() in ['gt_latitude', 'gt_lat']), None)
    gt_lon_col = next((c for c in df.columns if c.lower() in ['gt_longitude', 'gt_lon']), None)

    if lat_col is None or lon_col is None:
        print(f"  ❌ Could not find latitude/longitude columns in {csv_path}")
        print(f"     Columns found: {list(df.columns)}")
        return

    print(f"  Using columns: lat={lat_col}, lon={lon_col}, name={name_col}")

    results = []
    for idx, row in df.iterrows():
        name = row.get(name_col, f'Location_{idx}') if name_col else f'Location_{idx}'
        category = row.get(cat_col, 'unknown') if cat_col else 'unknown'
        address = row.get(addr_col, '') if addr_col else ''
        lat = row[lat_col]
        lon = row[lon_col]
        gt_lat = row.get(gt_lat_col) if gt_lat_col else None
        gt_lon = row.get(gt_lon_col) if gt_lon_col else None

        result = predict_and_display(
            name, category, address, lat, lon,
            gt_lat=gt_lat, gt_lon=gt_lon
        )
        if result:
            results.append(result)

    # Save batch results
    if results:
        results_df = pd.DataFrame(results)
        output_csv = OUTPUT_DIR / "batch_predictions.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\n{'=' * 70}")
        print(f"  BATCH COMPLETE: {len(results)} predictions")
        print(f"  Results saved to: {output_csv}")

        if 'improvement_m' in results_df.columns:
            print(f"\n  Summary:")
            print(f"    Mean original error:  {results_df['original_error_m'].mean():.2f}m")
            print(f"    Mean predicted error: {results_df['predicted_error_m'].mean():.2f}m")
            print(f"    Mean improvement:     {results_df['improvement_m'].mean():.2f}m")
            improved = (results_df['improvement_m'] > 0).sum()
            print(f"    Locations improved:   {improved}/{len(results_df)} "
                  f"({improved/len(results_df)*100:.0f}%)")

        print(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict corrected location using trained ML model"
    )
    parser.add_argument('--name', type=str, help='Place name')
    parser.add_argument('--category', type=str, help='Category (e.g. restaurant, hotel)')
    parser.add_argument('--address', type=str, help='Street address')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--confidence', type=float, default=0.9, help='Confidence score')
    parser.add_argument('--gt-lat', type=float, default=None, help='Ground truth latitude (optional)')
    parser.add_argument('--gt-lon', type=float, default=None, help='Ground truth longitude (optional)')
    parser.add_argument('--csv', type=str, default=None, help='CSV file for batch prediction')

    args = parser.parse_args()

    if args.csv:
        predict_batch(args.csv)
    elif args.lat and args.lon:
        predict_and_display(
            name=args.name or 'Unknown Place',
            category=args.category or 'unknown',
            address=args.address or '',
            lat=args.lat,
            lon=args.lon,
            confidence=args.confidence,
            gt_lat=args.gt_lat,
            gt_lon=args.gt_lon
        )
    else:
        # Default: run on test location from test_single_location.py
        print("\n  No arguments provided — running default test location")
        print("  Use --help for usage info\n")

        predict_and_display(
            name="La Belle Elaine's Bridal",
            category='bridal_shop',
            address='1550 Eastlake Ave E',
            lat=47.6337251,
            lon=-122.3253727,
            gt_lat=47.6336835,
            gt_lon=-122.3254237,
        )


if __name__ == "__main__":
    main()
