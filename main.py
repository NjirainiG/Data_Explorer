import streamlit as st
import requests
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from branca.element import Template, MacroElement

# Set page config for better performance
st.set_page_config(
    page_title="Kenya 2063 Ward Level Data Explorer",
    page_icon="📍",
    layout="wide"
)

# Cache the data loading function to avoid reloading on every interaction
@st.cache_data(ttl=3600, show_spinner="Loading ward data from Google Drive...")
def load_geojson_from_drive():
    """Load GeoJSON data from Google Drive with caching."""
    try:
        # Google Drive file ID for the ward stunting data (from secrets)
        file_id = st.secrets.get("GOOGLE_DRIVE_GEOJSON_FILE_ID")
        if not file_id:
            st.error("Google Drive file ID not configured in secrets.")
            return None
            
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Parse as GeoDataFrame
        gdf = gpd.read_file(response.text)
        
        # Ensure valid geometry and CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        
        return gdf
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def create_choropleth_map(gdf, indicator, centroid_y, centroid_x):
    """Create an optimized Folium choropleth map."""
    m = folium.Map(
        location=[centroid_y, centroid_x],
        zoom_start=6,  # Zoom out for Kenya overview
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Calculate min and max values
    min_val = gdf[indicator].min()
    max_val = gdf[indicator].max()
    
    # Create choropleth with simplified options
    choropleth = folium.Choropleth(
        geo_data=gdf,
        name='choropleth',
        data=gdf,
        columns=['ward', indicator],
        key_on='feature.properties.ward',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.05,
        line_weight=0.3,
        legend_name=f'{indicator}',
        highlight=False,  # Disable highlight for better performance
        bins=5,  # Reduced bins for faster rendering
        reset=True
    )
    
    # Add choropleth to map
    choropleth.add_to(m)
    
    # Get the style function for GeoJson (defined as a regular function for pickling)
    def style_function(feature):
        return {
            'weight': 0.3,
            'color': '#666666'
        }
    
    # Add tooltips with simplified style
    folium.GeoJson(
        gdf,
        name='Labels',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ward', indicator, 'county'],
            aliases=['Ward:', f'{indicator}:', 'County:'],
            localize=True,
            style="font-size: 11px;"
        )
    ).add_to(m)
    
    # Add minimal legend
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px;
        left: 50px;
        width: 120px;
        height: 70px;
        z-index:9999;
        font-size:12px;
        background: white;
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 5px;
        ">
        <div style="font-weight: bold; margin-bottom: 5px;">{{this.indicator}}</div>
        <div style="font-size: 11px;">High: {{this.max}}</div>
        <div style="font-size: 11px;">Low: {{this.min}}</div>
    </div>
    {% endmacro %}
    """
    
    macro = MacroElement()
    macro._template = Template(template)
    macro.max = f"{max_val:.1f}"
    macro.min = f"{min_val:.1f}"
    macro.indicator = indicator
    m.get_root().add_child(macro)
    
    return m

def main():
    st.title("📊 Kenya Ward-Level Stunting Data Explorer")
    
    # Load data with caching (spinner is handled by the cache decorator)
    gdf = load_geojson_from_drive()
    
    if gdf is None or gdf.empty:
        st.error("No data available. Please check your internet connection.")
        return
    
    # Display basic dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Wards", len(gdf))
    st.sidebar.metric("Total Counties", gdf['county'].nunique())
    
    # Get numeric columns for selection
    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')
    
    if not numeric_cols:
        st.warning("No numeric indicators found in the dataset.")
        st.dataframe(gdf.head())
        return
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Map Visualization", "📈 Data Analysis", "📥 Export Data"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Interactive Map")
            
            # Pre-calculate map center from bounds (faster than centroid)
            bounds = gdf.total_bounds
            centroid_y = (bounds[1] + bounds[3]) / 2
            centroid_x = (bounds[0] + bounds[2]) / 2
            
            # Indicator selection
            selected_indicator = st.selectbox(
                "Select indicator to visualize:",
                numeric_cols,
                key='map_indicator'
            )
            
            # Create and display map
            m = create_choropleth_map(gdf, selected_indicator, centroid_y, centroid_x)
            st_folium(m, width=700, height=500)
        
        with col2:
            st.subheader("Map Controls")
            
            # Quick statistics for selected indicator
            st.metric(
                f"Average {selected_indicator}",
                f"{gdf[selected_indicator].mean():.1f}"
            )
            st.metric(
                f"Highest {selected_indicator}",
                f"{gdf[selected_indicator].max():.1f}"
            )
            st.metric(
                f"Lowest {selected_indicator}",
                f"{gdf[selected_indicator].min():.1f}"
            )
            
            # Top wards for selected indicator
            st.write("**Top 5 Wards:**")
            top_wards = gdf.nlargest(5, selected_indicator)[['ward', selected_indicator]]
            for _, row in top_wards.iterrows():
                st.write(f"- {row['ward']}: {row[selected_indicator]:.1f}")
    
    with tab2:
        st.subheader("Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Summary Statistics**")
            # Cache summary statistics
            @st.cache_data(ttl=300)
            def get_summary_stats(_gdf, numeric_cols):
                return _gdf[numeric_cols].describe().T
            
            summary_stats = get_summary_stats(gdf, numeric_cols)
            st.dataframe(summary_stats.style.format("{:.2f}"))
        
        with col2:
            st.write("**County-Level Aggregation**")
            
            # Cache county aggregation
            @st.cache_data(ttl=300)
            def get_county_stats(_gdf, numeric_cols):
                return _gdf.groupby('county')[numeric_cols].mean()
            
            county_stats = get_county_stats(gdf, numeric_cols)
            st.dataframe(
                county_stats.style.format("{:.1f}"),
                use_container_width=True
            )
        
        # Correlation matrix (simplified) - only compute if requested
        if len(numeric_cols) > 1:
            if st.checkbox("Show correlation matrix", value=False):
                st.write("**Correlation Matrix**")
                correlation = gdf[numeric_cols].corr()
                st.dataframe(correlation.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
    
    with tab3:
        st.subheader("Export Data")
        
        # Filter options
        st.write("### Filter Options")
        
        # County filter
        all_counties = sorted(gdf['county'].unique())
        selected_counties = st.multiselect(
            "Select counties:",
            all_counties,
            default=all_counties[:3] if len(all_counties) > 3 else all_counties
        )
        
        # Numeric range filters
        st.write("### Range Filters")
        col1, col2 = st.columns(2)
        
        filter_expressions = []
        for i, col in enumerate(numeric_cols[:2]):  # Limit to 2 columns for UI simplicity
            col_container = col1 if i % 2 == 0 else col2
            with col_container:
                min_val = float(gdf[col].min())
                max_val = float(gdf[col].max())
                values = st.slider(
                    f"{col} range:",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"export_filter_{col}"
                )
                if values[0] > min_val or values[1] < max_val:
                    filter_expressions.append(f"({col} >= {values[0]}) & ({col} <= {values[1]})")
        
        # Apply filters
        filtered_gdf = gdf.copy()
        
        if selected_counties:
            filtered_gdf = filtered_gdf[filtered_gdf['county'].isin(selected_counties)]
        
        if filter_expressions:
            filter_query = " & ".join(filter_expressions)
            filtered_gdf = filtered_gdf.query(filter_query)
        
        st.write(f"**Filtered Results:** {len(filtered_gdf)} of {len(gdf)} wards")
        
        # Column selection for export
        all_columns = gdf.columns.tolist()
        export_columns = st.multiselect(
            "Select columns to export:",
            all_columns,
            default=['ward', 'county', 'subcounty'] + numeric_cols[:3]
        )
        
        if export_columns:
            export_df = filtered_gdf[export_columns]
            st.dataframe(export_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="kenya_ward_stunting_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create simplified GeoJSON for download - only when clicked
                if st.button("Generate GeoJSON for download"):
                    with st.spinner("Generating GeoJSON..."):
                        geojson_data = filtered_gdf[export_columns + ['geometry']].to_json()
                        st.download_button(
                            label="🗺️ Download as GeoJSON",
                            data=geojson_data,
                            file_name="kenya_ward_stunting_data.geojson",
                            mime="application/json"
                        )
                else:
                    st.info("Click 'Generate GeoJSON' to create download file")
    
    # Footer with dataset info
    st.sidebar.divider()
    st.sidebar.write("### About the Dataset")
    st.sidebar.write("""
    This dataset contains ward-level stunting rates 
    and related indicators across Kenya.
    
    **Indicators include:**
    - Stunting rates
    - Population data (2009)
    - County and subcounty information
    
    **Data Source:** Google Drive
    """)

if __name__ == "__main__":
    main()
