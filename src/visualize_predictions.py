import pandas as pd
import numpy as np
import folium
from folium import plugins
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

# Configuration
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "models"
INPUT_CSV = MODEL_DIR / "labeled_training_data.csv"
OUTPUT_MAP = DATA_DIR / "prediction_movement_map.html"



def get_color(improvement_m):
    """Color based on how much the prediction improved over original."""
    if improvement_m > 20:
        return '#00c853'      
    elif improvement_m > 5:
        return '#66bb6a'       
    elif improvement_m > 0:
        return '#a5d6a7'       
    elif improvement_m > -5:
        return '#ffab91'       
    else:
        return '#e53935'       


def get_error_color(error_m):
    """Color a marker based on absolute prediction error."""
    if error_m < 3:
        return '#00c853'    
    elif error_m < 8:
        return '#66bb6a'       
    elif error_m < 15:
        return '#ffb300'       
    elif error_m < 30:
        return '#ff7043'       
    else:
        return '#e53935'       


def main():
    
    print("  VISUALIZE ML PREDICTION MOVEMENT")
    

    # Load data
    print(f"\n  Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Matched: {len(df)} locations loaded")

    # Filter to rows with predictions
    df = df.dropna(subset=['pred_lat', 'pred_lon'])
    print(f"Matched: {len(df)} locations have predictions")

    # Calculate improvement
    df['improvement_m'] = df['original_error_m'] - df['pred_error_m']
    df['improved'] = df['improvement_m'] > 0

    # Stats
    improved_count = df['improved'].sum()
    total = len(df)
    pct = improved_count / total * 100

    print(f"\n  Summary:")
    print(f"    Mean original error:  {df['original_error_m'].mean():.2f}m")
    print(f"    Mean predicted error: {df['pred_error_m'].mean():.2f}m")
    print(f"    Mean improvement:     {df['improvement_m'].mean():.2f}m")
    print(f"    Locations improved:   {improved_count}/{total} ({pct:.0f}%)")
    print(f"    Median pred error:    {df['pred_error_m'].median():.2f}m")

    # ============================================================
    # Create map
    # ============================================================
    print(f"\n  Creating interactive map...")

    center_lat = df['gt_latitude'].mean()
    center_lon = df['gt_longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB positron',
        max_zoom=22
    )

    zoom_script = """
    <script>
    function findMap() {
        for (var key in window) {
            if (key.startsWith('map_') && window[key] instanceof L.Map) {
                return window[key];
            }
        }
        return null;
    }

    function zoomToPoint(lat, lon) {
        var map = findMap();
        if (map) {
            map.flyTo([lat, lon], 21, {duration: 1.0});
        } else {
            console.error("Leaflet map instance not found!");
        }
    }
    </script>
    """
    m.get_root().html.add_child(folium.Element(zoom_script))

    # Feature groups for toggling layers
    fg_arrows = folium.FeatureGroup(name="Movement Arrows", show=True)
    fg_original = folium.FeatureGroup(name="Original Positions (red)", show=False)
    fg_predicted_improved = folium.FeatureGroup(name="ML Predictions (Improved, blue)", show=True)
    fg_predicted_worsened = folium.FeatureGroup(name="ML Predictions (Worsened, blue)", show=False)
    fg_ground_truth = folium.FeatureGroup(name="Ground Truth (green)", show=False)
    fg_improved = folium.FeatureGroup(name="Improved (green arrows)", show=True)
    fg_worsened = folium.FeatureGroup(name="Worsened (red arrows)", show=True)

    for _, row in df.iterrows():
        orig_lat = row['original_lat']
        orig_lon = row['original_lon']
        pred_lat = row['pred_lat']
        pred_lon = row['pred_lon']
        gt_lat = row['gt_latitude']
        gt_lon = row['gt_longitude']

        name = row.get('place_name', 'Unknown')
        category = row.get('primary_category', '')
        orig_err = row['original_error_m']
        pred_err = row['pred_error_m']
        improvement = row['improvement_m']
        move_dist = row.get('pred_move_m', 0)

        # Popup HTML
        popup_html = f"""
        <div style="font-family: -apple-system, sans-serif; width: 260px;">
            <h4 style="margin: 0 0 4px 0; font-size: 13px;">{name[:40]}</h4>
            <p style="margin: 0; font-size: 11px; color: #888;">{category}</p>
            <div style="margin: 4px 0 0 0; font-size: 9px; color: #aaa; font-family: monospace; line-height: 1.2;">
                <div>ML: {pred_lat:.5f}, {pred_lon:.5f}</div>
                <div>GT: {gt_lat:.5f}, {gt_lon:.5f}</div>
            </div>
            <hr style="margin: 6px 0;">
            <table style="font-size: 11px; width: 100%;">
                <tr>
                    <td style="color: #e53935;">● Original error:</td>
                    <td style="text-align: right; font-weight: bold;">{orig_err:.1f}m</td>
                </tr>
                <tr>
                    <td style="color: #1e88e5;">★ ML error:</td>
                    <td style="text-align: right; font-weight: bold;">{pred_err:.1f}m</td>
                </tr>
                <tr>
                    <td>Move distance:</td>
                    <td style="text-align: right;">{move_dist:.1f}m</td>
                </tr>
                <tr style="border-top: 1px solid #eee;">
                    <td style="font-weight: bold;
                        color: {'#00c853' if improvement > 0 else '#e53935'};">
                        {'↓' if improvement > 0 else '↑'}
                        {'Improved' if improvement > 0 else 'Worse'}:
                    </td>
                    <td style="text-align: right; font-weight: bold;
                        color: {'#00c853' if improvement > 0 else '#e53935'};">
                        {abs(improvement):.1f}m
                    </td>
                </tr>
            </table>
            <div style="margin-top: 8px; text-align: center;">
                <a href="#" 
                   onclick="zoomToPoint({pred_lat}, {pred_lon}); return false;"
                   style="color: #1565c0; text-decoration: none; font-weight: bold; font-size: 11px;
                          border: 1px solid #1565c0; padding: 2px 6px; border-radius: 4px;">
                   🔍 Zoom Here
                </a>
            </div>
        </div>
        """

        popup = folium.Popup(popup_html, max_width=280)
        arrow_color = get_color(improvement)

        # --- Movement arrow: original → predicted ---
        target_fg = fg_improved if improvement > 0 else fg_worsened
        folium.PolyLine(
            locations=[[orig_lat, orig_lon], [pred_lat, pred_lon]],
            color=arrow_color, weight=2.5, opacity=0.7,
            popup=popup,
            tooltip=f"{name[:25]} | {'↓' if improvement > 0 else '↑'}{abs(improvement):.0f}m"
        ).add_to(target_fg)

        # --- Original position (red, small) ---
        folium.CircleMarker(
            location=[orig_lat, orig_lon],
            radius=3, color='#e53935', fill=True, fillColor='#e53935',
            fillOpacity=0.5, weight=1,
            tooltip=f"Original: {name[:25]}"
        ).add_to(fg_original)

        # --- ML Predicted position (blue star) ---
        target_pred_fg = fg_predicted_improved if improvement > 0 else fg_predicted_worsened
        error_color = get_error_color(pred_err)
        folium.CircleMarker(
            location=[pred_lat, pred_lon],
            radius=5, color=error_color, fill=True, fillColor=error_color,
            fillOpacity=0.8, weight=1.5,
            popup=popup,
            tooltip=f"ML: {name[:25]} ({pred_err:.1f}m error)"
        ).add_to(target_pred_fg)

        # --- Ground truth (green, small) ---
        folium.CircleMarker(
            location=[gt_lat, gt_lon],
            radius=3, color='#00c853', fill=True, fillColor='#00c853',
            fillOpacity=0.5, weight=1,
            tooltip=f"GT: {name[:25]}"
        ).add_to(fg_ground_truth)

    # Add feature groups
    for fg in [fg_original, fg_ground_truth, fg_predicted_improved, fg_predicted_worsened, fg_improved, fg_worsened]:
        fg.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # ============================================================
    # Add summary legend
    # ============================================================
    p10 = np.percentile(df['pred_error_m'], 10)
    p25 = np.percentile(df['pred_error_m'], 25)
    p50 = np.percentile(df['pred_error_m'], 50)
    p75 = np.percentile(df['pred_error_m'], 75)
    p90 = np.percentile(df['pred_error_m'], 90)

    worsened_count = total - improved_count

    legend_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; width: 300px;
         background: white; border: 2px solid #666; z-index: 9999;
         font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         font-size: 12px; padding: 14px; border-radius: 8px;
         box-shadow: 0 4px 12px rgba(0,0,0,0.15);">

         <h3 style="margin: 0 0 8px 0; font-size: 14px; color: #333;">
            ML Prediction Movement</h3>

         <div style="display: flex; gap: 8px; margin-bottom: 10px;">
            <div style="flex: 1; text-align: center; padding: 6px;
                 background: #e8f5e9; border-radius: 4px;">
                <div style="font-size: 18px; font-weight: bold; color: #2e7d32;">
                    {improved_count}</div>
                <div style="font-size: 10px; color: #4caf50;">improved</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 6px;
                 background: #fbe9e7; border-radius: 4px;">
                <div style="font-size: 18px; font-weight: bold; color: #c62828;">
                    {worsened_count}</div>
                <div style="font-size: 10px; color: #e53935;">worsened</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 6px;
                 background: #e3f2fd; border-radius: 4px;">
                <div style="font-size: 18px; font-weight: bold; color: #1565c0;">
                    {pct:.0f}%</div>
                <div style="font-size: 10px; color: #1e88e5;">win rate</div>
            </div>
         </div>

         <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 3px 0;">Mean original error</td>
                <td style="text-align: right; font-weight: bold;">
                    {df['original_error_m'].mean():.1f}m</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 3px 0; color: #1565c0;">Mean ML error</td>
                <td style="text-align: right; font-weight: bold; color: #1565c0;">
                    {df['pred_error_m'].mean():.1f}m</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 3px 0; color: #2e7d32;">Mean improvement</td>
                <td style="text-align: right; font-weight: bold; color: #2e7d32;">
                    {df['improvement_m'].mean():.1f}m</td>
            </tr>
         </table>

         <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #ddd;">
            <div style="font-size: 10px; font-weight: bold; color: #666; margin-bottom: 4px;">
                ML Error Percentiles</div>
            <div style="display: flex; font-size: 10px; color: #888;">
                <span style="flex:1;">P10</span>
                <span style="flex:1;">P25</span>
                <span style="flex:1; font-weight: bold; color: #333;">P50</span>
                <span style="flex:1;">P75</span>
                <span style="flex:1;">P90</span>
            </div>
            <div style="display: flex; font-size: 11px; font-weight: bold;">
                <span style="flex:1; color: #00c853;">{p10:.1f}m</span>
                <span style="flex:1; color: #66bb6a;">{p25:.1f}m</span>
                <span style="flex:1; color: #333;">{p50:.1f}m</span>
                <span style="flex:1; color: #ff7043;">{p75:.1f}m</span>
                <span style="flex:1; color: #e53935;">{p90:.1f}m</span>
            </div>
         </div>

         <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #ddd;
              font-size: 10px; color: #999;">
            <span style="color: #e53935;">●</span> Original &nbsp;
            <span style="color: #1e88e5;">●</span> ML Predicted &nbsp;
            <span style="color: #00c853;">●</span> Ground Truth<br>
            Arrow color = improvement (green) or regression (red)
         </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # ============================================================
    # Add clickable scrolling sidebar
    # ============================================================
    sorted_df = df.sort_values(by='improvement_m', ascending=True)
    sidebar_items = ""
    
    for _, row in sorted_df.iterrows():
        lat = row['pred_lat']
        lon = row['pred_lon']
        name = row.get('place_name', 'Unknown')
        category = row.get('primary_category', '')
        imp = row['improvement_m']
        
        color = '#e53935' if imp < 0 else '#00c853'
        symbol = '↑' if imp < 0 else '↓'
        bg = '#fbe9e7' if imp < 0 else '#e8f5e9'
        
        sidebar_items += f'''
        <div style="padding: 8px 12px; border-bottom: 1px solid #eee; background: {bg}; cursor: pointer;"
             onclick="zoomToPoint({lat}, {lon});"
             onmouseover="this.style.filter='brightness(0.95)'" onmouseout="this.style.filter='none'">
            <div style="font-weight: bold; font-size: 12px; margin-bottom: 2px;">{name[:30]}</div>
            <div style="font-size: 10px; color: #666; display: flex; justify-content: space-between;">
                <span>{category[:20]}</span>
                <span style="font-weight: bold; color: {color};">{symbol}{abs(imp):.1f}m</span>
            </div>
            <div style="font-size: 9px; color: #999; margin-top: 3px; font-family: monospace;">
                {lat:.5f}, {lon:.5f}
            </div>
        </div>
        '''

    sidebar_html = f'''
    <div style="position: fixed; top: 0; right: 0; width: 280px; height: 100vh;
         background: white; border-left: 1px solid #ccc; z-index: 9999;
         font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         display: flex; flex-direction: column;
         box-shadow: -2px 0 12px rgba(0,0,0,0.1);">
         
         <div style="padding: 15px; background: #fff; border-bottom: 1px solid #ddd;">
            <h3 style="margin: 0 0 5px 0; font-size: 14px; color: #333;">Location Explorer</h3>
            <p style="margin: 0; font-size: 11px; color: #666;">Click to zoom (Worst regressions at top)</p>
         </div>
         
         <div style="flex: 1; overflow-y: auto;">
            {sidebar_items}
         </div>
    </div>
    <style>
        .leaflet-control-container .leaflet-right {{
            right: 280px !important;
        }}
    </style>
    '''
    m.get_root().html.add_child(folium.Element(sidebar_html))

    # ============================================================
    # Add minimap
    # ============================================================
    plugins.MiniMap(toggle_display=True).add_to(m)

    # ============================================================
    # Save
    # ============================================================
    m.save(str(OUTPUT_MAP))
    print(f"\nMatched: Map saved to: {OUTPUT_MAP}")
    print(f"    Open in browser to explore!")

    # ============================================================
    # Print top 10 biggest improvements and top 10 worst regressions
    # ============================================================
    print(f"\n  {'─' * 60}")
    print(f"  TOP 10 BIGGEST IMPROVEMENTS:")
    print(f"  {'─' * 60}")
    top_improved = df.nlargest(10, 'improvement_m')
    for _, r in top_improved.iterrows():
        print(f"    {r['place_name'][:35]:<36s} "
              f"{r['original_error_m']:>6.1f}m → {r['pred_error_m']:>6.1f}m  "
              f"(↓{r['improvement_m']:.1f}m)")

    print(f"\n  {'─' * 60}")
    print(f"  TOP 10 WORST REGRESSIONS:")
    print(f"  {'─' * 60}")
    top_worsened = df.nsmallest(10, 'improvement_m')
    for _, r in top_worsened.iterrows():
        print(f"    {r['place_name'][:35]:<36s} "
              f"{r['original_error_m']:>6.1f}m → {r['pred_error_m']:>6.1f}m  "
              f"(↑{abs(r['improvement_m']):.1f}m)")

    # ============================================================
    # Category breakdown
    # ============================================================
    print(f"\n  {'─' * 60}")
    print(f"  PERFORMANCE BY PRIMARY CATEGORY (top 10):")
    print(f"  {'─' * 60}")
    cat_stats = df.groupby('primary_category').agg(
        count=('pred_error_m', 'size'),
        mean_orig=('original_error_m', 'mean'),
        mean_pred=('pred_error_m', 'mean'),
        mean_imp=('improvement_m', 'mean'),
    ).sort_values('count', ascending=False).head(10)

    print(f"    {'Category':<30s} {'N':>4s} {'Orig':>7s} {'ML':>7s} {'Imp':>7s}")
    for cat, r in cat_stats.iterrows():
        sign = "↓" if r['mean_imp'] > 0 else "↑"
        print(f"    {cat[:30]:<30s} {int(r['count']):>4d} "
              f"{r['mean_orig']:>6.1f}m {r['mean_pred']:>6.1f}m "
              f"{sign}{abs(r['mean_imp']):>5.1f}m")

    # ============================================================
    # Group type breakdown
    # ============================================================
    if 'category_type' in df.columns:
        print(f"\n  {'─' * 60}")
        print(f"  PERFORMANCE BY CATEGORY GROUP (High-Level):")
        print(f"  {'─' * 60}")
        group_stats = df.groupby('category_type').agg(
            count=('pred_error_m', 'size'),
            mean_orig=('original_error_m', 'mean'),
            mean_pred=('pred_error_m', 'mean'),
            mean_imp=('improvement_m', 'mean'),
        ).sort_values('mean_pred', ascending=True) # Sort by lowest error

        print(f"    {'Group':<20s} {'N':>4s} {'Orig':>7s} {'ML':>7s} {'Imp':>7s}")
        for cat, r in group_stats.iterrows():
            sign = "↓" if r['mean_imp'] > 0 else "↑"
            print(f"    {cat[:20]:<20s} {int(r['count']):>4d} "
                  f"{r['mean_orig']:>6.1f}m {r['mean_pred']:>6.1f}m "
                  f"{sign}{abs(r['mean_imp']):>5.1f}m")


    print(f"\n{'=' * 60}")
    print(f"  COMPLETE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
