import streamlit as st
import requests
import geopandas as gpd
import folium
import pandas as pd
import google.genai as genai
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
            return None, None
        
        # Initialize the client with the API key
        client = genai.Client(api_key=api_key)
        
        # Try the most likely models for the current API version
        known_models = [
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash',
            'gemini-2.0-flash-001',
            'gemini-2.5-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro'
        ]
        
        # Try each model
        for model_name in known_models:
            try:
                # Simple test to see if the model works
                response = client.models.generate_content(
                    model=model_name,
                    contents="Hello"
                )
                if response and hasattr(response, 'text'):
                    st.sidebar.success(f"✓ Using model: {model_name}")
                    return client, model_name
            except Exception as e:
                # Try the next model if this one fails
                continue
        
        # If no known model works, try fallback models
        st.warning("No known model worked. Trying fallback models...")
        fallback_models = [
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        for model_name in fallback_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Hello"
                )
                if response and hasattr(response, 'text'):
                    st.sidebar.success(f"✓ Using fallback model: {model_name}")
                    return client, model_name
            except:
                continue
        
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
        try:
            gdf = gpd.read_file(
                response.text,
                engine='pyogrio'  # Faster engine if available
            )
        except:
            # Fallback to default engine
            gdf = gpd.read_file(response.text)
        
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
    """Generate a comprehensive data summary for the AI agent with actual ward-level data."""
    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')
    
    # Identify stunting-related columns (case-insensitive)
    stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition', 'health', 'wasting', 'underweight']
    stunting_cols = []
    for col in numeric_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in stunting_keywords):
            stunting_cols.append(col)
    
    # Get county-specific data with actual ward-level stunting rates
    county_ward_data = {}
    for county in gdf['county'].unique():
        county_df = gdf[gdf['county'] == county]
        if stunting_cols:
            county_ward_data[county] = {}
            for col in stunting_cols[:3]:  # Top 3 stunting columns
                if col in county_df.columns:
                    # Get ward with highest stunting in this county
                    max_ward = county_df.loc[county_df[col].idxmax()] if not county_df.empty else None
                    min_ward = county_df.loc[county_df[col].idxmin()] if not county_df.empty else None
                    
                    county_ward_data[county][col] = {
                        'highest_ward': {
                            'ward': max_ward['ward'] if max_ward is not None else 'N/A',
                            'value': float(max_ward[col]) if max_ward is not None else 0.0,
                            'subcounty': max_ward['subcounty'] if max_ward is not None else 'N/A'
                        } if max_ward is not None else None,
                        'lowest_ward': {
                            'ward': min_ward['ward'] if min_ward is not None else 'N/A',
                            'value': float(min_ward[col]) if min_ward is not None else 0.0,
                            'subcounty': min_ward['subcounty'] if min_ward is not None else 'N/A'
                        } if min_ward is not None else None,
                        'county_average': float(county_df[col].mean()) if not county_df.empty else 0.0,
                        'ward_count': len(county_df)
                    }
    
    # Get Nakuru specific data (as an example for testing)
    nakuru_data = {}
    if 'Nakuru' in county_ward_data:
        nakuru_data = county_ward_data['Nakuru']
    
    summary = {
        "dataset_overview": {
            "data_granularity": "WARD-LEVEL (most granular administrative unit in Kenya)",
            "total_wards": len(gdf),
            "total_counties": gdf['county'].nunique(),
            "total_subcounties": gdf['subcounty'].nunique(),
            "columns": gdf.columns.tolist(),
            "numeric_columns": numeric_cols,
            "stunting_related_columns": stunting_cols,
            "has_ward_level_stunting_data": len(stunting_cols) > 0,
            "stunting_columns_found": stunting_cols
        },
        "summary_statistics": {},
        "ward_level_examples": {
            "nakuru_example": nakuru_data,
            "sample_ward_data": {}
        },
        "county_ward_analysis": county_ward_data,
        "top_bottom_wards": {}
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
        
        # Top 5 wards (highest values)
        top_5 = gdf.nlargest(5, col)[['ward', 'county', 'subcounty', col]]
        summary["top_bottom_wards"][col] = {
            "top_5_wards": top_5.to_dict('records'),
            "bottom_5_wards": gdf.nsmallest(5, col)[['ward', 'county', 'subcounty', col]].to_dict('records')
        }
    
    # Add sample ward data for a few counties
    sample_counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Kakamega']
    for county in sample_counties:
        if county in gdf['county'].values:
            county_wards = gdf[gdf['county'] == county]
            if not county_wards.empty and stunting_cols:
                for col in stunting_cols[:2]:  # First 2 stunting columns
                    if col in county_wards.columns:
                        sample_wards = county_wards[['ward', col, 'subcounty']].head(3).to_dict('records')
                        summary["ward_level_examples"]["sample_ward_data"][f"{county}_{col}"] = sample_wards
    
    # Add Nakuru specific ward data for testing
    if 'Nakuru' in gdf['county'].values:
        nakuru_wards = gdf[gdf['county'] == 'Nakuru']
        if not nakuru_wards.empty and stunting_cols:
            for col in stunting_cols[:3]:
                if col in nakuru_wards.columns:
                    # Get all wards in Nakuru with their stunting values
                    nakuru_all_wards = nakuru_wards[['ward', col, 'subcounty']].sort_values(col, ascending=False)
                    summary["ward_level_examples"]["nakuru_all_wards"] = nakuru_all_wards.to_dict('records')
                    break  # Just get the first stunting column
    
    return summary

def extract_specific_data_for_query(gdf, question):
    """Extract specific ward-level data based on the user's question."""
    extracted_data = {}
    
    # Check if question is about a specific county
    county_names = gdf['county'].unique().tolist()
    mentioned_counties = []
    
    for county in county_names:
        if county.lower() in question.lower():
            mentioned_counties.append(county)
    
    # If specific counties are mentioned, extract their ward-level data
    for county in mentioned_counties[:2]:  # Limit to first 2 mentioned counties
        county_df = gdf[gdf['county'] == county]
        
        # Get stunting columns
        numeric_cols = county_df.select_dtypes(include=['number']).columns.tolist()
        stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition', 'health']
        stunting_cols = []
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in stunting_keywords):
                stunting_cols.append(col)
        
        if stunting_cols:
            for col in stunting_cols[:3]:  # Top 3 stunting columns
                # Get top and bottom wards for this stunting indicator
                top_wards = county_df.nlargest(5, col)[['ward', col, 'subcounty']]
                bottom_wards = county_df.nsmallest(5, col)[['ward', col, 'subcounty']]
                
                extracted_data[f"{county}_{col}"] = {
                    "county": county,
                    "indicator": col,
                    "top_wards": top_wards.to_dict('records'),
                    "bottom_wards": bottom_wards.to_dict('records'),
                    "county_average": float(county_df[col].mean()),
                    "ward_count": len(county_df)
                }
    
    # If no specific counties mentioned but question is about stunting, provide overall data
    if not extracted_data and any(word in question.lower() for word in ['stunting', 'stunt', 'malnutrition']):
        numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
        stunting_cols = []
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['stunting', 'stunt', 'malnutrition']):
                stunting_cols.append(col)
        
        if stunting_cols:
            for col in stunting_cols[:2]:
                # National level top and bottom wards
                top_wards = gdf.nlargest(10, col)[['ward', 'county', 'subcounty', col]]
                bottom_wards = gdf.nsmallest(10, col)[['ward', 'county', 'subcounty', col]]
                
                extracted_data[f"national_{col}"] = {
                    "indicator": col,
                    "top_wards_national": top_wards.to_dict('records'),
                    "bottom_wards_national": bottom_wards.to_dict('records'),
                    "national_average": float(gdf[col].mean())
                }
    
    return extracted_data

def query_ai_agent(question, data_summary, client, model_name, gdf, chat_history=None):
    """Query the AI agent with the user's question and data context, including specific ward-level data."""
    # Extract specific ward-level data based on the question
    specific_data = extract_specific_data_for_query(gdf, question)
    
    # System prompt for the data scientist with economics background
    system_prompt = """
    You are a senior data scientist with an economics background specializing in public policy decision-making for Kenya's development goals (Kenya Vision 2063). You analyze ward-level data to provide actionable insights for policymakers.

    **CRITICAL DATA CONTEXT - READ CAREFULLY:**
    - The dataset contains **WARD-LEVEL STUNTING DATA** - this is the most granular administrative unit in Kenya
    - Stunting data is available at the ward level for ALL counties including Nairobi and Nakuru
    - The data includes {total_wards} wards across {total_counties} counties
    - You have access to ACTUAL WARD-LEVEL stunting rates - use them in your analysis
    - The dataset has ward-level stunting data: {has_stunting_data}
    - Stunting-related columns found in dataset: {stunting_columns}

    **AVAILABLE WARD-LEVEL DATA:**
    {specific_data}

    **DATASET SUMMARY:**
    {data_summary}

    **Your Expertise:**
    - Economic development and poverty reduction strategies
    - Public health and nutrition policy (stunting, malnutrition)
    - Regional development and resource allocation
    - Evidence-based policy recommendations
    - Spatial analysis and geographic disparities
    - Intra-county analysis using ward-level data

    **CRITICAL INSTRUCTIONS:**
    1. You MUST use the ACTUAL ward-level stunting data provided in the "AVAILABLE WARD-LEVEL DATA" section
    2. When asked about a specific county (like Nakuru), use the ward-level data for that county
    3. Provide specific ward names, their stunting rates, and subcounty information
    4. If ward-level data is provided for the question, you MUST reference it
    5. Do NOT say ward-level data is not available - it IS available in this dataset
    6. Use exact numbers from the data when available
    7. Conduct intra-county analysis to identify wards with highest stunting rates within counties
    8. Suggest targeted interventions for high-priority wards

    **Current Question:**
    {question}

    **IMPORTANT:** Your response MUST include:
    1. Specific ward names and their exact stunting rates from the data
    2. County and subcounty information for mentioned wards
    3. Comparative analysis within counties when relevant
    4. Policy recommendations based on the ward-level data
    5. Connection to Kenya Vision 2063 goals
    """
    
    # Extract key information from data summary for the prompt
    total_wards = data_summary.get("dataset_overview", {}).get("total_wards", "unknown")
    total_counties = data_summary.get("dataset_overview", {}).get("total_counties", "unknown")
    has_stunting_data = data_summary.get("dataset_overview", {}).get("has_ward_level_stunting_data", False)
    stunting_columns = data_summary.get("dataset_overview", {}).get("stunting_related_columns", [])
    
    # Prepare specific data context
    specific_data_context = json.dumps(specific_data, indent=2) if specific_data else "No specific data extracted for this query."
    
    # Prepare data summary context (truncated)
    data_summary_context = json.dumps({
        "dataset_overview": data_summary.get("dataset_overview", {}),
        "summary_statistics": {k: v for k, v in list(data_summary.get("summary_statistics", {}).items())[:5]},
        "ward_level_examples": data_summary.get("ward_level_examples", {})
    }, indent=2)
    
    # Prepare full prompt with injected context
    full_prompt = system_prompt.format(
        total_wards=total_wards,
        total_counties=total_counties,
        has_stunting_data=has_stunting_data,
        stunting_columns=", ".join(stunting_columns) if stunting_columns else "None identified",
        specific_data=specific_data_context,
        data_summary=data_summary_context,
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
            contents=full_prompt
        )
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Error querying AI agent: {str(e)}\n\nDebug info: Question was about ward-level stunting data which IS available in the dataset."

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
    
    # Display data info for debugging
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Information**")
    st.sidebar.write(f"Columns: {len(gdf.columns)}")
    st.sidebar.write(f"Sample columns: {list(gdf.columns[:5])}...")
    
    # Check for stunting columns
    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition']
    stunting_cols = []
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in stunting_keywords):
            stunting_cols.append(col)
    
    if stunting_cols:
        st.sidebar.success(f"✓ Found {len(stunting_cols)} stunting columns")
        st.sidebar.write("Stunting columns:", stunting_cols[:3])
    
    # Initialize Gemini AI
    ai_client, model_name = init_gemini()
    
    # Display basic dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Wards", len(gdf))
    st.sidebar.metric("Total Counties", gdf['county'].nunique())
    
    # Get numeric columns for selection
    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')
    
    if not numeric_cols:
        st.warning("No numeric indicators found.")
        st.dataframe(gdf.head())
        return
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map Visualization", "🤖 AI Policy Advisor", "📈 Data Analysis", "📥 Export Data"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Interactive Map")
            
            # Calculate map center
            bounds = gdf.total_bounds
            centroid_y = (bounds[1] + bounds[3]) / 2
            centroid_x = (bounds[0] + bounds[2]) / 2
            
            # Indicator selection - prioritize stunting columns
            available_indicators = []
            if stunting_cols:
                available_indicators = stunting_cols + [col for col in numeric_cols if col not in stunting_cols]
            else:
                available_indicators = numeric_cols
            
            selected_indicator = st.selectbox(
                "Select indicator to visualize:",
                available_indicators,
                key='map_indicator'
            )
            
            # Create and display map
            m = create_choropleth_map(gdf, selected_indicator, centroid_y, centroid_x)
            
            # Display map with error handling
            try:
                st_folium(m, width=700, height=500, key=f"map_{selected_indicator}")
            except Exception as e:
                st.error(f"Map rendering error: {str(e)}")
        
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
            top_wards = gdf.nlargest(5, selected_indicator)[['ward', selected_indicator, 'county']]
            for _, row in top_wards.iterrows():
                st.write(f"- {row['ward']} ({row['county']}): {row[selected_indicator]:.1f}")
    
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
        
        # Show sample ward-level stunting data
        if stunting_cols:
            st.write("### Ward-Level Stunting Data Sample")
            sample_col = stunting_cols[0]
            st.write(f"**Sample of ward-level {sample_col} data:**")
            sample_data = gdf[['ward', 'county', 'subcounty', sample_col]].sort_values(sample_col, ascending=False).head(10)
            st.dataframe(sample_data.style.format({sample_col: "{:.2f}"}))
    
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
        **Ask questions about the ward-level stunting data to get insights from our AI Data Scientist.**
        
        ***Example questions about ward-level data:***
        - Which ward in Nakuru has the highest stunting rate?
        - What are the top 5 wards with highest stunting rates nationally?
        - Compare stunting rates between wards in Nairobi County
        - Which subcounty in Mombasa has the worst stunting problem?
        - What is the stunting rate in [specific ward name]?
        - Analyze ward-level disparities in stunting within Kakamega County
        - Recommend targeted interventions for high-stunting wards in Kisumu
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
            if prompt := st.chat_input("Ask about ward-level stunting data..."):
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing ward-level data..."):
                        response = query_ai_agent(
                            prompt, 
                            st.session_state.data_summary, 
                            ai_client,
                            model_name,
                            gdf,  # Pass the actual GeoDataFrame
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
    This dataset contains **ward-level stunting rates** 
    and related indicators across Kenya.
    
    **Key Features:**
    - Ward-level stunting data (most granular level)
    - County and subcounty information
    - Geographic boundaries for mapping
    - Multiple stunting/nutrition indicators
    
    **Data Source:** Google Drive
    
    **AI Features:** Powered by Google Gemini
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the logs for details.")

