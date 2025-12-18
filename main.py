import streamlit as st
import requests
import geopandas as gpd
import folium
import pandas as pd
import google.generativeai as genai
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
import json
import matplotlib.cm as cm
import matplotlib.colors as colors

# Constants for optimization
MAX_CHAT_HISTORY = 20
SIMPLIFICATION_TOLERANCE = 0.001

# Set page config for better performance
st.set_page_config(
    page_title="Kenya 2063 Ward Level Data Explorer",
    page_icon="📍",
    layout="wide"
)

def get_color(value, min_val, max_val):
    """Get color for choropleth using matplotlib."""
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    cmap = cm.YlOrRd
    rgba = cmap(norm(value))
    return colors.to_hex(rgba)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    """Initialize Gemini API with caching and error handling."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.warning("GEMINI_API_KEY not found in secrets. AI features will be disabled.")
            return None
        
        # Configure with new API
        client = genai.Client(api_key=api_key)
        
        # List available models and find a suitable one
        try:
            # Get list of available models
            models = client.list_models()
            
            # Try these models in order of preference
            preferred_models = [
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash',
                'gemini-2.0-flash-001',
                'gemini-2.5-flash',
                'gemini-1.5-flash',
                'gemini-1.5-pro'
            ]
            
            # Filter to available models
            available_model_names = [model.name for model in models]
            
            # Try each preferred model that's available
            for model_name in preferred_models:
                if any(model_name in name for name in available_model_names):
                    try:
                        # Test with a simple prompt
                        response = client.models.generate_content(
                            model=model_name,
                            contents=["Hello"]
                        )
                        if response and response.text:
                            st.sidebar.success(f"✓ Using model: {model_name}")
                            return client, model_name
                    except Exception as e:
                        continue
            
            # If no preferred model works, try the first available Gemini model
            gemini_models = [name for name in available_model_names if 'gemini' in name.lower()]
            if gemini_models:
                model_name = gemini_models[0].split('/')[-1]  # Remove 'models/' prefix
                response = client.models.generate_content(
                    model=model_name,
                    contents=["Hello"]
                )
                if response and response.text:
                    st.sidebar.success(f"✓ Using available model: {model_name}")
                    return client, model_name
        
        except Exception as e:
            st.error(f"Error finding model: {str(e)}")
        
        st.error("No compatible Gemini model found. Please check your API key and permissions.")
        return None, None
        
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None, None

# Cache the data loading function to avoid reloading on every interaction
@st.cache_data(ttl=3600, show_spinner="Loading ward data...")
def load_geojson_from_drive():
    """Load GeoJSON data with robust error handling and optimization."""
    try:
        # Google Drive file ID for the ward stunting data (from secrets)
        file_id = st.secrets.get("GOOGLE_DRIVE_GEOJSON_FILE_ID")
        if not file_id:
            st.error("Google Drive file ID not configured in secrets.")
            return gpd.GeoDataFrame()
        
        # Create download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Configure session for better performance
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Request with timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = session.get(download_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if response is valid
        if not response.text.strip():
            st.error("Received empty response from Google Drive")
            return gpd.GeoDataFrame()
        
        # Load GeoDataFrame with optimization
        gdf = gpd.read_file(
            response.text,
            engine='pyogrio'  # Faster engine if available
        )
        
        # Validate the GeoDataFrame
        if gdf.empty:
            st.warning("Loaded GeoDataFrame is empty")
            return gdf
        
        # Ensure valid geometry and CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        
        # Simplify geometries for better performance
        gdf['geometry'] = gdf.geometry.simplify(SIMPLIFICATION_TOLERANCE)
        
        # Optimize memory usage by downcasting numeric columns
        for col in gdf.select_dtypes(include=['number']).columns:
            if gdf[col].dtype in ['float64', 'int64']:
                gdf[col] = pd.to_numeric(gdf[col], downcast='float')
        
        st.sidebar.success(f"✓ Loaded {len(gdf)} wards")
        return gdf
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        st.info("Check file permissions and ensure it's publicly accessible.")
        return gpd.GeoDataFrame()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return gpd.GeoDataFrame()

def create_choropleth_map(gdf, indicator, centroid_y, centroid_x):
    """Create an optimized Folium choropleth map."""
    m = folium.Map(
        location=[centroid_y, centroid_x],
        zoom_start=6,
        tiles='cartodbpositron',  # Lighter tiles
        control_scale=True,
        prefer_canvas=True  # Better performance for many polygons
    )
    
    # Calculate min and max values
    min_val = gdf[indicator].min()
    max_val = gdf[indicator].max()
    
    # Create style function
    def style_function(feature):
        value = feature['properties'][indicator]
        return {
            'fillColor': get_color(value, min_val, max_val),
            'color': '#666666',
            'weight': 0.3,
            'fillOpacity': 0.7
        }
    
    # Add GeoJson with styling and tooltips
    folium.GeoJson(
        gdf,
        name='choropleth',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ward', indicator, 'county'],
            aliases=['Ward:', f'{indicator}:', 'County:'],
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

def get_data_summary(gdf):
    """Generate a comprehensive data summary for the AI agent."""
    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')
    
    # Identify stunting-related columns
    stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition']
    stunting_cols = []
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in stunting_keywords):
            stunting_cols.append(col)
    
    summary = {
        "dataset_overview": {
            "data_granularity": "WARD-LEVEL",
            "total_wards": len(gdf),
            "total_counties": gdf['county'].nunique(),
            "total_subcounties": gdf['subcounty'].nunique(),
            "columns": gdf.columns.tolist(),
            "numeric_columns": numeric_cols,
            "stunting_related_columns": stunting_cols,
            "has_ward_level_stunting_data": len(stunting_cols) > 0
        },
        "summary_statistics": {},
        "top_performers": {},
        "bottom_performers": {},
        "regional_insights": {
            "county_level": {},
            "ward_level_examples": {}
        }
    }
    
    # Calculate summary statistics for numeric columns
    for col in numeric_cols:
        summary["summary_statistics"][col] = {
            "mean": float(gdf[col].mean()),
            "median": float(gdf[col].median()),
            "min": float(gdf[col].min()),
            "max": float(gdf[col].max()),
            "std": float(gdf[col].std()),
            "ward_level_available": True
        }
        
        # Top 5 performers
        top_5 = gdf.nlargest(5, col)[['ward', 'county', col]]
        summary["top_performers"][col] = top_5.to_dict('records')
        
        # Bottom 5 performers
        bottom_5 = gdf.nsmallest(5, col)[['ward', 'county', col]]
        summary["bottom_performers"][col] = bottom_5.to_dict('records')
    
    # County-level aggregation
    county_stats = gdf.groupby('county')[numeric_cols].mean().reset_index()
    summary["regional_insights"]["county_level"] = county_stats.to_dict('records')
    
    # Ward-level examples for Nairobi
    nairobi_wards = gdf[gdf['county'].str.contains('Nairobi', case=False, na=False)]
    if not nairobi_wards.empty and stunting_cols:
        for col in stunting_cols[:2]:
            nairobi_top = nairobi_wards.nlargest(3, col)[['ward', col]]
            summary["regional_insights"]["ward_level_examples"][f"nairobi_{col}"] = nairobi_top.to_dict('records')
    
    return summary

def query_ai_agent(question, data_summary, client, model_name, chat_history=None):
    """Query the AI agent with the user's question and data context."""
    # System prompt for the data scientist
    system_prompt = """
    You are a senior data scientist with an economics background specializing in public policy decision-making for Kenya's development goals (Kenya Vision 2063). You analyze ward-level data to provide actionable insights for policymakers.

    **CRITICAL DATA CONTEXT - READ CAREFULLY:**
    - The dataset contains **WARD-LEVEL stunting data** - this is the most granular administrative unit in Kenya
    - Stunting data is available at the ward level for ALL counties including Nairobi
    - The data includes {total_wards} wards across {total_counties} counties
    - Ward-level stunting data enables intra-county analysis to identify hotspots within counties
    - The dataset has ward-level stunting data: {has_stunting_data}
    - Stunting-related columns in dataset: {stunting_columns}

    **Your Expertise:**
    - Economic development and poverty reduction strategies
    - Public health and nutrition policy (stunting, malnutrition)
    - Regional development and resource allocation
    - Evidence-based policy recommendations
    - Spatial analysis and geographic disparities
    - Intra-county analysis using ward-level data

    **Available Data Context:**
    {data_context}

    **Guidelines for Responses:**
    1. ALWAYS acknowledge that ward-level stunting data is available when discussing data granularity
    2. Use specific ward-level examples from the data (top/bottom performers by ward)
    3. Conduct intra-county analysis to identify wards with highest stunting rates within counties
    4. Consider economic implications and policy trade-offs
    5. Highlight geographic disparities and regional patterns at both county AND ward levels
    6. Suggest targeted interventions for high-priority wards (not just counties)
    7. Connect findings to Kenya Vision 2063 goals
    8. Be concise but comprehensive in your analysis
    9. Use specific numbers from the data when available
    10. NEVER suggest that ward-level stunting data is missing - it is available in this dataset

    **Current Question:**
    {question}
    """
    
    # Extract key information from data summary
    total_wards = data_summary.get("dataset_overview", {}).get("total_wards", "unknown")
    total_counties = data_summary.get("dataset_overview", {}).get("total_counties", "unknown")
    has_stunting_data = data_summary.get("dataset_overview", {}).get("has_ward_level_stunting_data", False)
    stunting_columns = data_summary.get("dataset_overview", {}).get("stunting_related_columns", [])
    
    # Prepare data context (truncate if too long)
    data_context = json.dumps(data_summary, indent=2)
    if len(data_context) > 14000:
        data_context = data_context[:14000] + "\n... (truncated)"
    
    # Prepare full prompt
    full_prompt = system_prompt.format(
        total_wards=total_wards,
        total_counties=total_counties,
        has_stunting_data=has_stunting_data,
        stunting_columns=", ".join(stunting_columns) if stunting_columns else "None identified",
        data_context=data_context,
        question=question
    )
    
    try:
        # Include chat history if available
        if chat_history and len(chat_history) > 0:
            conversation_context = "\nPrevious conversation:\n"
            for msg in chat_history[-5:]:
                conversation_context += f"{msg['role']}: {msg['content']}\n"
            full_prompt = conversation_context + "\n" + full_prompt
        
        # Generate response using new API
        response = client.models.generate_content(
            model=model_name,
            contents=[full_prompt]
        )
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Error querying AI agent: {str(e)}"

def main():
    st.title("📊 Kenya Ward-Level Stunting Data Explorer with AI Policy Advisor")
    
    # Add simple health check indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("**App Status:** ✅ Running")
    
    # Load data with caching
    with st.spinner("Loading ward-level data..."):
        gdf = load_geojson_from_drive()
    
    if gdf is None or gdf.empty:
        st.error("No data available. Please check your data source and connection.")
        return
    
    # Initialize Gemini AI
    ai_client, model_name = init_gemini()
    
    # Display basic dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Wards", len(gdf))
    st.sidebar.metric("Total Counties", gdf['county'].nunique())
    
    # Get numeric columns for selection
    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')
    
    if not numeric_cols:
        st.warning("No numeric indicators found.")
        st.dataframe(gdf.head())
        return
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map Visualization", "📈 Data Analysis", "📥 Export Data", "🤖 AI Policy Advisor"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Interactive Map")
            
            # Calculate map center
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
            
            # Display map with error handling
            try:
                st_folium(m, width=700, height=500, key=f"map_{selected_indicator}")
            except Exception as e:
                st.error(f"Map rendering error: {str(e)}")
                # Fallback: show static map image or data table
        
        with col2:
            st.subheader("Map Controls")
            
            # Quick statistics
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
            
            # Top wards
            st.write("**Top 5 Wards:**")
            top_wards = gdf.nlargest(5, selected_indicator)[['ward', selected_indicator]]
            for _, row in top_wards.iterrows():
                st.write(f"- {row['ward']}: {row[selected_indicator]:.1f}")
    
    with tab2:
        st.subheader("Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Summary Statistics**")
            @st.cache_data(ttl=300)
            def get_summary_stats(_gdf, _numeric_cols):
                return _gdf[_numeric_cols].describe().T
            
            summary_stats = get_summary_stats(gdf, numeric_cols)
            st.dataframe(summary_stats.style.format("{:.2f}"))
        
        with col2:
            st.write("**County-Level Aggregation**")
            
            @st.cache_data(ttl=300)
            def get_county_stats(_gdf, _numeric_cols):
                return _gdf.groupby('county')[_numeric_cols].mean()
            
            county_stats = get_county_stats(gdf, numeric_cols)
            st.dataframe(
                county_stats.style.format("{:.1f}"),
                use_container_width=True
            )
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            if st.checkbox("Show correlation matrix", value=False):
                st.write("**Correlation Matrix**")
                @st.cache_data(ttl=300)
                def get_correlation(_gdf, _numeric_cols):
                    return _gdf[_numeric_cols].corr()
                
                correlation = get_correlation(gdf, numeric_cols)
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
        for i, col in enumerate(numeric_cols[:2]):
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
                # Generate GeoJSON on demand
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
    
    with tab4:
        st.subheader("🤖 AI Policy Advisor")
        st.markdown("""
        **Ask questions about the data to get insights from our AI Data Scientist with economics background.**
        
        *Example questions:*
        - Which counties have the highest stunting rates?
        - What is the relationship between population and stunting rates?
        - Recommend policy interventions for high-stunting areas
        - Analyze regional disparities in stunting rates
        - How does this data relate to Kenya Vision 2063 goals?
        """)
        
        # Initialize session state for chat
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'data_summary' not in st.session_state:
            st.session_state.data_summary = get_data_summary(gdf)
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if ai_client and model_name:
            if prompt := st.chat_input("Ask a question about the data..."):
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data and formulating policy insights..."):
                        response = query_ai_agent(
                            prompt, 
                            st.session_state.data_summary, 
                            ai_client,
                            model_name,
                            st.session_state.chat_history
                        )
                        st.markdown(response)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Trim chat history if too long
                if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
        else:
            st.warning("⚠️ AI features are currently disabled.")
            st.info("To enable AI features, add your Gemini API key to `.streamlit/secrets.toml`:")
            st.code("GEMINI_API_KEY = 'your-api-key-here'")
    
    # Footer with dataset info
    st.sidebar.divider()
    st.sidebar.write("### About the Dataset")
    st.sidebar.write("""
    This dataset contains ward-level stunting rates 
    and related indicators across Kenya.
    
    **Indicators include:**
    - Stunting rates
    - Population data
    - County and subcounty information
    
    **Data Source:** Google Drive
    
    **AI Features:** Powered by Google Gemini
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the logs for details.")

