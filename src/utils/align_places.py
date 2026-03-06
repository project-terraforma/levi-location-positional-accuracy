"""
Place Alignment
===============

This module refines ML-predicted location pins by aligning them. It groups 
predictions by their associated building polygon and perfectly aligns points 
(e.g. stores in a strip mall) that share the same wall segment and fall within 
a similar depth threshold.

Points are projected onto a normal vector facing outward from the building's 
centroid, ensuring unified orientations along building facades.

Returns a DataFrame with 'aligned_lat' and 'aligned_lon' columns added.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from shapely import wkt
from shapely.geometry import Point, LineString
import warnings
warnings.filterwarnings('ignore')
def align_predictions(df, depth_threshold_m=20.0):
    req_cols = ['pred_lat', 'pred_lon', 'building_wkt', 'centroid_lat', 'centroid_lon']
    for col in req_cols:
        if col not in df.columns:
            df['aligned_lat'] = df.get('pred_lat', df.get('gt_latitude', df.get('original_lat')))
            df['aligned_lon'] = df.get('pred_lon', df.get('gt_longitude', df.get('original_lon')))
            return df
            
    df['aligned_lat'] = df['pred_lat']
    df['aligned_lon'] = df['pred_lon']
    df['was_aligned'] = False
    
    LAT_TO_M = 111000
    
    # Group by valid buildings only
    valid_bldgs = df[df['building_wkt'].notna() & (df['building_wkt'] != 'None')]
    groups = valid_bldgs.groupby('building_wkt')
    
    for b_wkt, group_df in groups:
        # We need at least 2 points in a building to "align" them to each other
        if len(group_df) < 2:
            continue
            
        c_lat = group_df.iloc[0].get('centroid_lat')
        c_lon = group_df.iloc[0].get('centroid_lon')
        if pd.isna(c_lat) or pd.isna(c_lon):
            continue
            
        LON_TO_M = 111000 * np.cos(np.radians(c_lat))
        
        poly = None
        try:
            poly = wkt.loads(b_wkt)
        except:
            pass
            
        if poly is None or not hasattr(poly, 'exterior'):
            continue
            
        coords = list(poly.exterior.coords)
        segments = []
        for i in range(len(coords) - 1):
            segments.append(LineString([coords[i], coords[i+1]]))
            
        # 1. Assign each point to its nearest outer wall segment
        segment_groups = {} # segment_idx -> list of (row_idx, point_lon, point_lat)
        
        for idx, row in group_df.iterrows():
            pred_pt = Point(row['pred_lon'], row['pred_lat'])
            
            min_dist_m = float('inf')
            best_seg_idx = -1
            
            for i, segment in enumerate(segments):
                dist_deg = segment.distance(pred_pt)
                dist_m = dist_deg * 111000 # rough meters
                if dist_m < min_dist_m:
                    min_dist_m = dist_m
                    best_seg_idx = i
                    
            # 2. Skip points inside the center of the building
            if min_dist_m > 10.0 or best_seg_idx == -1:
                continue
                
            if best_seg_idx not in segment_groups:
                segment_groups[best_seg_idx] = []
            segment_groups[best_seg_idx].append((idx, row['pred_lon'], row['pred_lat']))
            
        # 3. Align points that share the same wall segment
        for seg_idx, points in segment_groups.items():
            if len(points) < 2:
                continue # Needs at least 2 points along the same wall to form a line
                
            # Wall vector in pseudo-meters
            p1 = coords[seg_idx]
            p2 = coords[seg_idx + 1]
            wx = (p2[0] - p1[0]) * LON_TO_M
            wy = (p2[1] - p1[1]) * LAT_TO_M
            
            # Normal vector orthogonal to the wall
            nx, ny = -wy, wx
            
            mag = np.sqrt(nx**2 + ny**2)
            if mag == 0:
                continue
            nx /= mag
            ny /= mag
            
            # We want the normal pointing Outward from the centroid
            cx = (coords[seg_idx][0] - c_lon) * LON_TO_M
            cy = (coords[seg_idx][1] - c_lat) * LAT_TO_M
            
            # If dot product is negative, normal is pointing inward, so flip it
            if (nx * cx + ny * cy) < 0:
                nx = -nx
                ny = -ny
                
            depths = []
            for item in points:
                idx, p_lon, p_lat = item
                px = (p_lon - c_lon) * LON_TO_M
                py = (p_lat - c_lat) * LAT_TO_M
                depth = px * nx + py * ny
                depths.append((item, depth, px, py))
                
            depths.sort(key=lambda x: x[1])
            clusters = []
            current_cluster = [depths[0]]
            for i in range(1, len(depths)):
                if depths[i][1] - current_cluster[-1][1] <= depth_threshold_m:
                    current_cluster.append(depths[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [depths[i]]
            clusters.append(current_cluster)
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Favor the point closest to the street (maximum outward depth)
                    target_depth = np.max([x[1] for x in cluster])
            
                    for x in cluster:
                        item, orig_depth, px, py = x
                        idx = item[0]
                        depth_diff = target_depth - orig_depth
                        
                        new_px = px + depth_diff * nx
                        new_py = py + depth_diff * ny
                        
                        new_lon = c_lon + (new_px / LON_TO_M)
                        new_lat = c_lat + (new_py / LAT_TO_M)
                        
                        pt = Point(new_lon, new_lat)
                        if not poly.contains(pt):
                            from shapely.ops import nearest_points
                            geom_on_poly, _ = nearest_points(poly, pt)
                            new_lon = geom_on_poly.x
                            new_lat = geom_on_poly.y
                            
                        df.loc[idx, 'aligned_lat'] = new_lat
                        df.loc[idx, 'aligned_lon'] = new_lon
                        df.loc[idx, 'was_aligned'] = True

    return df
