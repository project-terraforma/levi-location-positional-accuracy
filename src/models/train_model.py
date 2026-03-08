"""
Model Training & Evaluation
===========================

This script trains a gradient boosting regression model to predict the optimal
offset (Δlat, Δlon) for correcting pin locations. It uses conflated building 
and street features (OSM + Overture) to learn how to place a point.

Workflow:
  1. Loads conflated features (e.g. from get_conflated_repositioning.py)
  2. Builds context and anchor-based feature representations
  3. Evaluates oracle bounds (best possible baseline)
  4. Trains a direct multi-output regressor with Huber loss to resist outliers
  5. Saves the trained model and feature information to be used during inference
"""
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import json
import warnings
warnings.filterwarnings('ignore')


# Configuration
DATA_DIR = Path("data")
OSM_FEATURES_CSV = DATA_DIR / "osm_features" / "osm_repositioning_features.csv"
CONFLATED_FEATURES_CSV = DATA_DIR / "conflated_features" / "conflated_repositioning_features.csv"
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Anchor strategies from get_osm_repositioning.py
STRATEGY_NAMES = [
    'centroid', 'nearest_edge',
    'street_1_facing', 'street_2_facing', 'street_3_facing',
    'parking_facing',
    'south_edge', 'north_edge', 'east_edge', 'west_edge'
]


# Utility Functions
def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points (Haversine)."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def weighted_blend(lats, lons, weights):
    """Compute weighted average position from anchor points and weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalise to sum=1
    blended_lat = np.dot(lats, weights)
    blended_lon = np.dot(lons, weights)
    return blended_lat, blended_lon


def clamp_prediction(pred_lat, pred_lon, row, available_strategies, buffer_deg=0.0005):
    """
    Clamp a predicted lat/lon to stay within the bounding box of the
    available anchor points (with a small buffer ~50m).  If the prediction
    is still >100m from the centroid, fall back to nearest_edge.
    """
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

    # Gather anchor coords
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

    category_type = row.get('category_type', 'other')
    is_outdoor = (category_type == 'outdoor')
    is_storage = (category_type == 'self_storage_facility')
    building_wkt = row.get('building_wkt')
    
    if not is_outdoor and not is_storage and pd.notna(building_wkt):
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
                eps = 0.01
                p1 = boundary.interpolate(max(0, proj_dist - eps))
                p2 = boundary.interpolate(min(boundary.length, proj_dist + eps))
                tx, ty = p2.x - p1.x, p2.y - p1.y
                centroid = poly.centroid
                dx_c = centroid.x - clamped_lon
                dy_c = centroid.y - clamped_lat
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


def trimmed_mean(values, pct=5):
    """Mean after removing the top `pct`% of values (outlier-robust)."""
    arr = np.array(values)
    cutoff = np.percentile(arr, 100 - pct)
    return arr[arr <= cutoff].mean()


# Step 1: Load and Prepare Data
def load_data():
    """Load OSM features and ground truth."""
    
    print("STEP 1: LOADING DATA")
    

    #osm_df = pd.read_csv(OSM_FEATURES_CSV)
    osm_df = pd.read_csv(CONFLATED_FEATURES_CSV)
    print(f"Matched: OSM features: {len(osm_df)} rows, {len(osm_df.columns)} columns")

    # Keep only matched buildings 
    matched = osm_df[osm_df['matched_building'] == True].copy()
    print(f"Matched: Matched locations (with building): {len(matched)}")

    return matched


# Step 2: Build Feature Matrix
def build_features(df):
    """
    Build two feature sets:
      A) context_features — category, address, building meta, street meta
      B) anchor_coords     — lat/lon of every available strategy point
    """
    print("STEP 2: FEATURE ENGINEERING")

    out = df.copy()

    out['confidence'] = pd.to_numeric(out['confidence'], errors='coerce').fillna(0.5)
    out['building_area_sqm'] = pd.to_numeric(out['building_area_sqm'], errors='coerce').fillna(0)
    out['log_building_area'] = np.log1p(out['building_area_sqm'])
    out['building_distance_m'] = pd.to_numeric(out['building_distance_m'], errors='coerce').fillna(50)
    out['num_streets'] = pd.to_numeric(out['num_streets'], errors='coerce').fillna(0).astype(int)
    out['has_parking'] = out['has_parking'].astype(int)
    out['num_strategies'] = pd.to_numeric(out['num_strategies'], errors='coerce').fillna(6).astype(int)

    def count_surrounding(val):
        if pd.isna(val) or val == '':
            return 0
        try:
            return len(json.loads(val))
        except Exception:
            return 0
            
    out['num_surrounding_buildings'] = out.get('surrounding_buildings', pd.Series(dtype=object)).apply(count_surrounding)

    # Building match method (one-hot encode the new types)
    # Types: address_match_exact, address_match_fuzzy, distance_based, etc.
    match_dummies = pd.get_dummies(out['building_match_method'], prefix='match')
    out = pd.concat([out, match_dummies], axis=1)
    match_cols = match_dummies.columns.tolist()

    # Street features
    for i in range(1, 4):
        col = f'street_{i}_match_method'
        out[f'street_{i}_addr_match'] = (out[col] == 'address_match').astype(int) if col in out.columns else 0
        dist_col = f'street_{i}_distance_m'
        out[dist_col] = pd.to_numeric(out.get(dist_col, pd.Series(200)), errors='coerce').fillna(200)

    out['any_street_addr_match'] = (
        out.get('street_1_addr_match', 0) |
        out.get('street_2_addr_match', 0) |
        out.get('street_3_addr_match', 0)
    ).astype(int)

    # Address features
    out['has_address'] = out['formatted_address'].notna().astype(int)
    out['address_length'] = out['formatted_address'].fillna('').str.len()
    out['address_has_number'] = out['formatted_address'].fillna('').str.contains(r'\d', regex=True).astype(int)

    # Category
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
    out['category_type'] = out['primary_category'].map(category_type_map).fillna('other')
    cat_dummies = pd.get_dummies(out['category_type'], prefix='cat')
    out = pd.concat([out, cat_dummies], axis=1)

    anchor_lat_cols = []
    anchor_lon_cols = []
    available_strategies = []
    for s in STRATEGY_NAMES:
        lat_c = f'{s}_lat'
        lon_c = f'{s}_lon'
        if lat_c in out.columns:
            anchor_lat_cols.append(lat_c)
            anchor_lon_cols.append(lon_c)
            available_strategies.append(s)

    # Relative offsets: strategy position minus original position
    for s in available_strategies:
        lat_c = f'{s}_lat'
        lon_c = f'{s}_lon'
        out[f'{s}_dlat'] = out[lat_c] - out['original_lat']
        out[f'{s}_dlon'] = out[lon_c] - out['original_lon']

    dlat_cols = [f'{s}_dlat' for s in available_strategies]
    dlon_cols = [f'{s}_dlon' for s in available_strategies]

    # Inter-anchor distances (pairwise)
    from itertools import combinations
    pair_cols = []
    for s1, s2 in combinations(available_strategies[:5], 2):  # top 5 to limit features
        col = f'spread_{s1}_{s2}'
        out[col] = np.sqrt(
            (out[f'{s1}_dlat'] - out[f'{s2}_dlat'])**2 +
            (out[f'{s1}_dlon'] - out[f'{s2}_dlon'])**2
        )
        pair_cols.append(col)

    context_features = [
        'confidence', 'building_area_sqm', 'log_building_area',
        'building_distance_m', 
        'num_streets', 'has_parking', 'num_strategies', 'num_surrounding_buildings',
        'street_1_addr_match', 'street_2_addr_match', 'street_3_addr_match',
        'any_street_addr_match',
        'street_1_distance_m', 'street_2_distance_m', 'street_3_distance_m',
        'has_address', 'address_length', 'address_has_number',
    ] + [c for c in out.columns if c.startswith('cat_')] + match_cols

    anchor_offset_features = dlat_cols + dlon_cols
    shape_features = pair_cols

    all_features = context_features + anchor_offset_features + shape_features

    # Fill NaNs in feature columns
    for col in all_features:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    print(f"Matched: Context features:       {len(context_features)}")
    print(f"Matched: Anchor offset features: {len(anchor_offset_features)}")
    print(f"Matched: Shape features:         {len(shape_features)}")
    print(f"Matched: Total features:         {len(all_features)}")
    print(f"Matched: Available strategies:    {available_strategies}")

    return out, all_features, context_features, available_strategies


# Step 3: Oracle Analysis — What's the best we can achieve?
def oracle_analysis(df, available_strategies):
    """Compute oracle bounds: best single strategy, best blend of two, etc."""
    print("Matched: ORACLE & BASELINE ANALYSIS")

    results = {}

    # 1. Original position error
    orig_dists = []
    for _, row in df.iterrows():
        d = haversine_distance(row['original_lat'], row['original_lon'],
                               row['gt_latitude'], row['gt_longitude'])
        orig_dists.append(d)
    df['original_error_m'] = orig_dists
    results['original_position'] = np.mean(orig_dists)

    # 2. Per-strategy "always use X" baselines
    for s in available_strategies:
        lat_c, lon_c = f'{s}_lat', f'{s}_lon'
        valid = df[df[lat_c].notna() & df[lon_c].notna()]
        if len(valid) == 0:
            continue
        dists = [haversine_distance(r[lat_c], r[lon_c], r['gt_latitude'], r['gt_longitude'])
                 for _, r in valid.iterrows()]
        results[f'always_{s}'] = np.mean(dists)

    # 3. Oracle best single strategy per location
    best_dists = []
    best_strategies = []
    for _, row in df.iterrows():
        min_dist = float('inf')
        min_s = 'centroid'
        for s in available_strategies:
            lat_c, lon_c = f'{s}_lat', f'{s}_lon'
            if pd.notna(row.get(lat_c)) and pd.notna(row.get(lon_c)):
                d = haversine_distance(row[lat_c], row[lon_c],
                                       row['gt_latitude'], row['gt_longitude'])
                if d < min_dist:
                    min_dist = d
                    min_s = s
        best_dists.append(min_dist)
        best_strategies.append(min_s)
    df['oracle_best_strategy'] = best_strategies
    df['oracle_best_dist'] = best_dists
    results['oracle_best_single'] = np.mean(best_dists)

    # 4. Oracle best BLEND of two strategies per location
    #    For each location, try all pairs with alpha in [0,1] and find minimum
    from itertools import combinations
    blend_dists = []
    for _, row in df.iterrows():
        min_dist = float('inf')
        gt_lat, gt_lon = row['gt_latitude'], row['gt_longitude']

        for s1, s2 in combinations(available_strategies, 2):
            lat1 = row.get(f'{s1}_lat')
            lon1 = row.get(f'{s1}_lon')
            lat2 = row.get(f'{s2}_lat')
            lon2 = row.get(f'{s2}_lon')

            if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
                continue

            # Search alpha from 0 to 1 in steps of 0.05
            for alpha in np.arange(0, 1.01, 0.05):
                blend_lat = alpha * lat1 + (1 - alpha) * lat2
                blend_lon = alpha * lon1 + (1 - alpha) * lon2
                d = haversine_distance(blend_lat, blend_lon, gt_lat, gt_lon)
                if d < min_dist:
                    min_dist = d

        blend_dists.append(min_dist)

    df['oracle_blend_dist'] = blend_dists
    results['oracle_best_blend_2'] = np.mean(blend_dists)

    # Print results sorted worst → best
    print(f"\n  {'Method':<35s} {'Mean Dist to GT':>15s}")
    print(f"  {'─'*35} {'─'*15}")
    for method, dist in sorted(results.items(), key=lambda x: x[1], reverse=True):
        tag = ""
        if method == 'original_position':
            tag = "  ← baseline"
        elif method == 'oracle_best_single':
            tag = "  ← best single"
        elif method == 'oracle_best_blend_2':
            tag = "  ← best pair blend"
        print(f"  {method:<35s} {dist:>12.2f}m{tag}")

    max_imp = results['original_position'] - results['oracle_best_blend_2']
    pct = max_imp / results['original_position'] * 100
    print(f"\nMatched: Max possible improvement (blend): {max_imp:.2f}m ({pct:.1f}%)")

    return results, df


# Step 4: Train Direct Regression — predict Δlat, Δlon
def train_direct_regressor(df, all_features, available_strategies, context_features):
    """
    Model that takes context + all anchor offsets and directly predicts the
    optimal Δlat and Δlon from the original position.  This lets the model
    implicitly learn any combination of the anchor points.

    Uses Huber loss (robust to outliers) and clamps predictions to the
    bounding box of anchor points.
    """
    print("STEP 4: TRAINING DIRECT REGRESSOR (with clamping + Huber loss)")

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import KFold
        from sklearn.multioutput import MultiOutputRegressor
        import joblib
    except ImportError:
        print("  Warning: scikit-learn not installed. pip install scikit-learn")
        return None, None

    # Target: offset from original to ground truth
    # Target: offset from original to ground truth
    df = df.copy()
    # TRAIN ON ALL DATA (including outliers)
    
    df['target_dlat'] = df['gt_latitude'] - df['original_lat']
    df['target_dlon'] = df['gt_longitude'] - df['original_lon']

    X = df[all_features].values
    y = df[['target_dlat', 'target_dlon']].values

    print(f"  Training data: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Target: Δlat, Δlon from original to ground truth")

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_dists_raw = []      # before clamping
    cv_dists_clamped = []   # after clamping

    # Store per-sample predictions so we can save them
    pred_lats = np.full(len(df), np.nan)
    pred_lons = np.full(len(df), np.nan)
    pred_errors = np.full(len(df), np.nan)
    fold_means = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Huber loss is more robust to outlier targets than squared error
        reg = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            loss='huber', alpha=0.9,
            min_samples_split=5, min_samples_leaf=3,
            subsample=0.8, random_state=42
        ))
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        fold_dists = []
        for i in range(len(test_idx)):
            row = df.iloc[test_idx[i]]
            orig_lat = row['original_lat']
            orig_lon = row['original_lon']
            raw_lat = orig_lat + y_pred[i, 0]
            raw_lon = orig_lon + y_pred[i, 1]
            gt_lat = row['gt_latitude']
            gt_lon = row['gt_longitude']

            # Raw distance (before clamping)
            d_raw = haversine_distance(raw_lat, raw_lon, gt_lat, gt_lon)
            cv_dists_raw.append(d_raw)

            # Clamp prediction to anchor bounding box + fallback
            clamped_lat, clamped_lon = clamp_prediction(
                raw_lat, raw_lon, row, available_strategies
            )
            d_clamped = haversine_distance(clamped_lat, clamped_lon, gt_lat, gt_lon)
            fold_dists.append(d_clamped)
            cv_dists_clamped.append(d_clamped)

            # Store per-sample prediction
            pred_lats[test_idx[i]] = clamped_lat
            pred_lons[test_idx[i]] = clamped_lon
            pred_errors[test_idx[i]] = d_clamped

        fold_means.append(np.mean(fold_dists))
        print(f"    Fold {fold}: mean={np.mean(fold_dists):.2f}m  "
              f"median={np.median(fold_dists):.2f}m  "
              f"(raw mean={np.mean([cv_dists_raw[-len(fold_dists)+j] for j in range(len(fold_dists))]):.2f}m)")

    print(f"\n  Cross-validated Direct Regressor (CLAMPED):")
    print(f"    Mean distance to GT:         {np.mean(cv_dists_clamped):.2f}m")
    print(f"    Median distance to GT:       {np.median(cv_dists_clamped):.2f}m")
    print(f"    Trimmed mean (drop top 5%):  {trimmed_mean(cv_dists_clamped):.2f}m")
    print(f"    Std across folds:            {np.std(fold_means):.2f}m")
    print(f"  (Before clamping — raw mean:   {np.mean(cv_dists_raw):.2f}m)")

    # Build per-sample prediction arrays to return
    move_dists = []
    for i in range(len(df)):
        if np.isnan(pred_lats[i]):
            move_dists.append(np.nan)
        else:
            move_dists.append(
                haversine_distance(df.iloc[i]['original_lat'], df.iloc[i]['original_lon'],
                                   pred_lats[i], pred_lons[i]))
    pred_move = np.array(move_dists)

    # Train final model on all data
    final_reg = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        loss='huber', alpha=0.9,
        min_samples_split=5, min_samples_leaf=3,
        subsample=0.8, random_state=42
    ))
    final_reg.fit(X, y)

    # Feature importance (average across lat/lon estimators)
    imp_lat = final_reg.estimators_[0].feature_importances_
    imp_lon = final_reg.estimators_[1].feature_importances_
    avg_imp = (imp_lat + imp_lon) / 2
    imp_series = pd.Series(avg_imp, index=all_features).sort_values(ascending=False)

    print(f"\n  Top 15 feature importances:")
    for feat, imp in imp_series.head(15).items():
        print(f"    {feat:40s}: {imp:.4f}")

    # Save
    model_path = MODEL_DIR / "direct_regressor.joblib"
    joblib.dump({
        'model': final_reg,
        'feature_cols': all_features,
        'available_strategies': available_strategies,
        'cv_mean_dist': np.mean(cv_dists_clamped),
        'cv_median_dist': np.median(cv_dists_clamped),
        'cv_trimmed_mean': trimmed_mean(cv_dists_clamped),
    }, model_path)
    print(f"\nMatched: Model saved to: {model_path}")

    return final_reg, np.mean(cv_dists_clamped), {
        'pred_lat': pred_lats,
        'pred_lon': pred_lons,
        'pred_error_m': pred_errors,
        'pred_move_m': pred_move,
    }


# Step 5: Final Comparison
def print_final_comparison(baseline_results, direct_cv):
    """Print a clean comparison of all methods."""
    print("  FINAL COMPARISON — Mean Distance to Ground Truth")

    all_results = dict(baseline_results)
    if direct_cv is not None:
        all_results['ML_direct_regressor'] = direct_cv


    print(f"\n  {'Method':<35s} {'Distance':>10s} {'Improvement':>12s}")
    print(f"  {'─'*35} {'─'*10} {'─'*12}")

    orig = all_results.get('original_position', 0)
    for method, dist in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        delta = orig - dist
        sign = "+" if delta < 0 else "-"
        tag = ""
        if method == 'original_position':
            tag = " ← baseline"
        elif 'ML_' in method:
            tag = " ★ ML"
        elif 'oracle' in method:
            tag = " ← oracle"
        print(f"  {method:<35s} {dist:>8.2f}m  {sign}{abs(delta):>8.2f}m{tag}")

    # ML improvement summary
    if direct_cv:
        imp = orig - direct_cv
        pct = imp / orig * 100 if orig > 0 else 0
        print(f"\n  ★ ML Direct Regressor")
        print(f"    Improvement over original: {imp:.2f}m ({pct:.1f}%)")
        ne = all_results.get('always_nearest_edge', orig)
        if direct_cv < ne:
            print(f"    Also beats 'always nearest-edge' by {ne - direct_cv:.2f}m")
    

# Main Pipeline
def main():
    print("  ML PIPELINE: Learned Position Blending from OSM Strategies")
    
    # 1. Load
    df = load_data()

    # 2. Features
    featured_df, all_features, context_features, available_strategies = build_features(df)

    # 3. Oracle & baselines
    baseline_results, featured_df = oracle_analysis(featured_df, available_strategies)

    # 4. Direct regressor (context + anchor offsets → Δlat, Δlon)
    direct_reg, direct_cv, direct_preds = train_direct_regressor(
        featured_df, all_features, available_strategies, context_features
    )

    # Store per-sample predictions on featured_df
    if direct_preds:
        for col, vals in direct_preds.items():
            featured_df[col] = vals

    # 5. Final comparison
    print_final_comparison(baseline_results, direct_cv)

    # Save metadata
    meta = {
        'all_features': all_features,
        'context_features': context_features,
        'available_strategies': available_strategies,
        'n_training_samples': len(featured_df),
        'baseline_results': {k: float(v) for k, v in baseline_results.items()},
        'ml_direct_cv_mean': float(direct_cv) if direct_cv else None,

    }
    with open(MODEL_DIR / "model_info.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # Save labelled training data (now includes model predictions)
    save_cols = ['id', 'place_name', 'primary_category', 'category_type', 'formatted_address',
                 'original_lat', 'original_lon', 'gt_latitude', 'gt_longitude',
                 'original_error_m', 'oracle_best_strategy', 'oracle_best_dist',
                 'oracle_blend_dist',
                 'pred_lat', 'pred_lon', 'pred_error_m', 'pred_move_m']
    save_cols = [c for c in save_cols if c in featured_df.columns]
    featured_df[save_cols].to_csv(MODEL_DIR / "labeled_training_data.csv", index=False)

    
    print("  PIPELINE COMPLETE!")
    
    print(f"\n  Outputs in {MODEL_DIR}/:")
    print(f"    ├── direct_regressor.joblib    (context+anchors → lat/lon)")
    print(f"    ├── labeled_training_data.csv   (training data)")
    print(f"    └── model_info.json            (features & results)")
    

if __name__ == "__main__":
    main()
