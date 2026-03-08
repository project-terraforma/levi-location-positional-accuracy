import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from pathlib import Path
import numpy as np

# Configuration
DATA_DIR = Path("data")
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
OUTPUT_MAP = DATA_DIR / "offset_analysis_map.html"

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def main():
    print("=" * 60)
    print("DATA OFFSET ANALYSIS")
    print("=" * 60)
    
    # Load ground truth (manually labeled correct positions)
    print(f"\nLoading ground truth data from {GROUND_TRUTH_CSV}...")
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    print(f"✓ Loaded {len(gt_df)} ground truth locations")
    
    # Parse coordinates from WKT in ground truth
    def parse_wkt(wkt):
        if pd.isna(wkt):
            return None, None
        coords = wkt.replace('POINT (', '').replace(')', '').split()
        return float(coords[0]), float(coords[1])
    
    # Extract ground truth coordinates
    gt_df[['gt_lon', 'gt_lat']] = gt_df['WKT'].apply(lambda x: pd.Series(parse_wkt(x)))
    
    # Extract original coordinates from original_geometry_wkt
    gt_df[['orig_lon', 'orig_lat']] = gt_df['original_geometry_wkt'].apply(lambda x: pd.Series(parse_wkt(x)))
    
    # Calculate offset distances
    print("\nCalculating offset distances...")
    distances = []
    for idx, row in gt_df.iterrows():
        if pd.notna(row['orig_lat']) and pd.notna(row['gt_lat']):
            dist = calculate_distance(
                row['orig_lat'], row['orig_lon'],
                row['gt_lat'], row['gt_lon']
            )
            distances.append(dist)
        else:
            distances.append(None)
    
    gt_df['offset_distance_m'] = distances
    
    # Print statistics
    print("\n" + "=" * 60)
    print("OFFSET STATISTICS")
    print("=" * 60)
    
    valid_distances = [d for d in distances if d is not None]
    
    print(f"\nTotal locations: {len(gt_df)}")
    print(f"Valid offsets calculated: {len(valid_distances)}")
    
    if len(valid_distances) > 0:
        print(f"\nOffset Distance (meters):")
        print(f"  Mean:   {np.mean(valid_distances):.2f}m")
        print(f"  Median: {np.median(valid_distances):.2f}m")
        print(f"  Min:    {np.min(valid_distances):.2f}m")
        print(f"  Max:    {np.max(valid_distances):.2f}m")
        print(f"  Std:    {np.std(valid_distances):.2f}m")
        
        # Percentiles
        print(f"\nPercentiles:")
        print(f"  25th: {np.percentile(valid_distances, 25):.2f}m")
        print(f"  50th: {np.percentile(valid_distances, 50):.2f}m")
        print(f"  75th: {np.percentile(valid_distances, 75):.2f}m")
        print(f"  90th: {np.percentile(valid_distances, 90):.2f}m")
        print(f"  95th: {np.percentile(valid_distances, 95):.2f}m")
        print(f"  99th: {np.percentile(valid_distances, 99):.2f}m")
        
        # Distance ranges
        print(f"\nDistance Ranges:")
        print(f"  0-5m:     {sum(1 for d in valid_distances if d <= 5)} ({sum(1 for d in valid_distances if d <= 5)/len(valid_distances)*100:.1f}%)")
        print(f"  5-10m:    {sum(1 for d in valid_distances if 5 < d <= 10)} ({sum(1 for d in valid_distances if 5 < d <= 10)/len(valid_distances)*100:.1f}%)")
        print(f"  10-20m:   {sum(1 for d in valid_distances if 10 < d <= 20)} ({sum(1 for d in valid_distances if 10 < d <= 20)/len(valid_distances)*100:.1f}%)")
        print(f"  20-50m:   {sum(1 for d in valid_distances if 20 < d <= 50)} ({sum(1 for d in valid_distances if 20 < d <= 50)/len(valid_distances)*100:.1f}%)")
        print(f"  50-100m:  {sum(1 for d in valid_distances if 50 < d <= 100)} ({sum(1 for d in valid_distances if 50 < d <= 100)/len(valid_distances)*100:.1f}%)")
        print(f"  >100m:    {sum(1 for d in valid_distances if d > 100)} ({sum(1 for d in valid_distances if d > 100)/len(valid_distances)*100:.1f}%)")
    
    # Create interactive map
    print("\n" + "=" * 60)
    print("CREATING INTERACTIVE MAP")
    print("=" * 60)
    
    # Filter to valid locations
    valid_df = gt_df[gt_df['offset_distance_m'].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid locations to map!")
        return
    
    # Center map on mean location
    center_lat = valid_df['gt_lat'].mean()
    center_lon = valid_df['gt_lon'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # Add each location
    for idx, row in valid_df.iterrows():
        # Create feature group for this location
        fg = folium.FeatureGroup(name=f"{row['Name'][:30]}... ({row['offset_distance_m']:.1f}m)")
        
        # Add original point (red)
        folium.CircleMarker(
            location=[row['orig_lat'], row['orig_lon']],
            radius=3,
            popup=folium.Popup(
                f"<b>Original (Overture)</b><br>"
                f"{row['Name']}<br>"
                f"{row['Address']}<br>"
                f"Offset: {row['offset_distance_m']:.2f}m",
                max_width=300
            ),
            color='#e53935',
            fill=True,
            fillColor='#e53935',
            fillOpacity=0.7,
            weight=1
        ).add_to(fg)
        
        # Add ground truth point (blue)
        folium.CircleMarker(
            location=[row['gt_lat'], row['gt_lon']],
            radius=3,
            popup=folium.Popup(
                f"<b>Ground Truth (Corrected)</b><br>"
                f"{row['Name']}<br>"
                f"{row['Address']}<br>"
                f"Offset: {row['offset_distance_m']:.2f}m",
                max_width=300
            ),
            color='#1e88e5',
            fill=True,
            fillColor='#1e88e5',
            fillOpacity=0.7,
            weight=1
        ).add_to(fg)
        
        # Draw line connecting them
        folium.PolyLine(
            locations=[
                [row['orig_lat'], row['orig_lon']],
                [row['gt_lat'], row['gt_lon']]
            ],
            color='#9e9e9e',
            weight=1.5,
            opacity=0.6,
            popup=f"Offset: {row['offset_distance_m']:.2f}m"
        ).add_to(fg)
        
        fg.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; width: 250px; 
         background: white; border: 2px solid #666; z-index: 9999;
         font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         font-size: 12px; padding: 14px; border-radius: 8px;
         box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
         <h3 style="margin:0 0 8px 0; font-size:14px; color:#333;">Data Offset Analysis</h3>
         <p style="margin:2px;"><span style="color:#e53935;">●</span> Original (Overture)</p>
         <p style="margin:2px;"><span style="color:#1e88e5;">●</span> Ground Truth (Corrected)</p>
         <p style="margin:2px;"><span style="color:#9e9e9e;">━</span> Offset Line</p>
         <hr style="margin:8px 0; border:0; border-top:1px solid #ddd;">
         <p style="margin:2px 0 4px 0; font-size:11px; font-weight:bold; color:#666;">Statistics:</p>
         <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 2px 0;">Mean offset</td>
                <td style="text-align: right; font-weight: bold;">{np.mean(valid_distances):.1f}m</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 2px 0;">Median offset</td>
                <td style="text-align: right; font-weight: bold;">{np.median(valid_distances):.1f}m</td>
            </tr>
            <tr>
                <td style="padding: 2px 0;">Max offset</td>
                <td style="text-align: right; font-weight: bold;">{np.max(valid_distances):.1f}m</td>
            </tr>
         </table>
         <div style="margin-top: 8px; font-size: 10px; color: #999;">
            Toggle layers top right to isolate points
         </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(str(OUTPUT_MAP))
    print(f"✓ Map saved to: {OUTPUT_MAP}")
    
    # Export offset data to CSV
    output_csv = DATA_DIR / "offset_analysis.csv"
    export_cols = ['id', 'Name', 'Category', 'Address', 'Confidence Score',
                   'orig_lat', 'orig_lon', 'gt_lat', 'gt_lon', 'offset_distance_m']
    valid_df[export_cols].to_csv(output_csv, index=False)
    print(f"✓ Offset data saved to: {output_csv}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nOpen {OUTPUT_MAP} to view the interactive map")
    print("Red markers = Original Overture data")
    print("Green markers = Ground truth (manually corrected)")
    print("Blue lines = Offset distance")

if __name__ == "__main__":
    main()
