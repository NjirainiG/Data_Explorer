from google import genai  # FIX 1: was `import google.genai as genai` — correct import for google-genai SDK
import streamlit as st
import requests
import geopandas as gpd
import folium
import pandas as pd
from streamlit_folium import st_folium
from branca.colormap import LinearColormap  # FIX 3: replaces broken MacroElement legend
import json
import matplotlib.cm as cm
import matplotlib.colors as colors
import io

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
    if pd.isna(value):
        return '#808080'  # Gray for missing values
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

        known_models = [
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash',
            'gemini-2.0-flash-001',
            'gemini-2.5-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro'
        ]

        for model_name in known_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Hello"
                )
                if response and hasattr(response, 'text'):
                    st.sidebar.success(f"✓ Using model: {model_name}")
                    return client, model_name
            except Exception:
                continue

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
            except Exception:
                continue

        st.error("No compatible Gemini model found. Please check your API key and permissions.")
        return None, None

    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None, None


def download_file_from_google_drive(file_id):
    """Download file from Google Drive with proper handling of large files."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True, timeout=30)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True, timeout=30)
            break

    return response


@st.cache_data(ttl=3600, show_spinner="Loading ward data...")
def load_geojson_from_drive():
    """Load GeoJSON data with robust error handling and optimization."""
    try:
        file_id = st.secrets.get("GOOGLE_DRIVE_GEOJSON_FILE_ID")
        if not file_id:
            st.error("Google Drive file ID not configured in secrets.")
            return gpd.GeoDataFrame()

        response = download_file_from_google_drive(file_id)
        response.raise_for_status()

        content = response.content
        if not content:
            st.error("Received empty response from Google Drive")
            return gpd.GeoDataFrame()

        try:
            content_text = content.decode('utf-8')
            json.loads(content_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            st.error(f"Invalid GeoJSON format: {str(e)}")
            return gpd.GeoDataFrame()

        try:
            gdf = gpd.read_file(io.StringIO(content_text))
        except Exception as e:
            st.error(f"Error parsing GeoJSON: {str(e)}")
            return gpd.GeoDataFrame()

        if gdf.empty:
            st.warning("Loaded GeoDataFrame is empty")
            return gdf

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)

        gdf['geometry'] = gdf['geometry'].make_valid()
        gdf['geometry'] = gdf.geometry.simplify(SIMPLIFICATION_TOLERANCE)

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
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return gpd.GeoDataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def create_choropleth_map(_gdf, indicator, centroid_y, centroid_x):
    """Create an optimized Folium choropleth map with caching."""
    try:
        m = folium.Map(
            location=[centroid_y, centroid_x],
            zoom_start=6,
            tiles='cartodbpositron',
            control_scale=True,
            prefer_canvas=True
        )

        valid_values = _gdf[indicator].dropna()
        if len(valid_values) == 0:
            st.warning(f"No valid values found for {indicator}")
            return m

        min_val = float(valid_values.min())
        max_val = float(valid_values.max())

        geojson_data = json.loads(_gdf.to_json())

        def style_function(feature):
            try:
                props = feature.get('properties', {})
                value = props.get(indicator)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return {'fillColor': '#808080', 'color': '#666666', 'weight': 0.3, 'fillOpacity': 0.5}
                return {'fillColor': get_color(value, min_val, max_val), 'color': '#666666', 'weight': 0.3, 'fillOpacity': 0.7}
            except Exception:
                return {'fillColor': '#808080', 'color': '#666666', 'weight': 0.3, 'fillOpacity': 0.5}

        tooltip_fields = []
        tooltip_aliases = []
        for field, alias in [('ward', 'Ward:'), (indicator, f'{indicator}:'), ('county', 'County:')]:
            if field in _gdf.columns:
                tooltip_fields.append(field)
                tooltip_aliases.append(alias)

        folium.GeoJson(
            geojson_data,
            name='choropleth',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                style="font-size: 11px;"
            )
        ).add_to(m)

        # FIX 3: Use LinearColormap instead of broken MacroElement template
        colormap = LinearColormap(
            colors=['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
            vmin=min_val,
            vmax=max_val,
            caption=indicator
        )
        colormap.add_to(m)

        return m

    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return folium.Map(location=[centroid_y, centroid_x], zoom_start=6, tiles='cartodbpositron')


def get_data_summary(gdf):
    """Generate a comprehensive data summary for the AI agent."""
    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')

    stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition', 'health', 'wasting', 'underweight']
    stunting_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in stunting_keywords)]

    county_ward_data = {}
    if 'county' in gdf.columns:
        for county in gdf['county'].unique():
            county_df = gdf[gdf['county'] == county]
            if stunting_cols:
                county_ward_data[county] = {}
                for col in stunting_cols[:3]:
                    if col in county_df.columns:
                        valid = county_df[col].dropna()
                        if valid.empty:
                            continue
                        # FIX 4: guard idxmax/idxmin against all-NaN columns
                        max_idx = county_df[col].idxmax(skipna=True)
                        min_idx = county_df[col].idxmin(skipna=True)
                        max_ward = county_df.loc[max_idx]
                        min_ward = county_df.loc[min_idx]

                        county_ward_data[county][col] = {
                            'highest_ward': {
                                'ward': max_ward.get('ward', 'N/A'),
                                'value': float(max_ward[col]),
                                'subcounty': max_ward.get('subcounty', 'N/A')
                            },
                            'lowest_ward': {
                                'ward': min_ward.get('ward', 'N/A'),
                                'value': float(min_ward[col]),
                                'subcounty': min_ward.get('subcounty', 'N/A')
                            },
                            'county_average': float(county_df[col].mean()),
                            'ward_count': len(county_df)
                        }

    summary = {
        "dataset_overview": {
            "data_granularity": "WARD-LEVEL",
            "total_wards": len(gdf),
            "total_counties": gdf['county'].nunique() if 'county' in gdf.columns else 0,
            "total_subcounties": gdf['subcounty'].nunique() if 'subcounty' in gdf.columns else 0,
            "columns": gdf.columns.tolist(),
            "numeric_columns": numeric_cols,
            "stunting_related_columns": stunting_cols,
            "has_ward_level_stunting_data": len(stunting_cols) > 0,
            "stunting_columns_found": stunting_cols
        },
        "summary_statistics": {},
        "ward_level_examples": {"sample_ward_data": {}},
        "county_ward_analysis": county_ward_data,
        "top_bottom_wards": {}
    }

    for col in numeric_cols:
        summary["summary_statistics"][col] = {
            "mean": float(gdf[col].mean()),
            "median": float(gdf[col].median()),
            "min": float(gdf[col].min()),
            "max": float(gdf[col].max()),
            "std": float(gdf[col].std()),
        }
        top_5 = gdf.nlargest(5, col)[['ward', 'county', 'subcounty', col]]
        summary["top_bottom_wards"][col] = {
            "top_5_wards": top_5.to_dict('records'),
            "bottom_5_wards": gdf.nsmallest(5, col)[['ward', 'county', 'subcounty', col]].to_dict('records')
        }

    return summary


def extract_specific_data_for_query(gdf, question):
    """Extract specific ward-level data based on the user's question."""
    extracted_data = {}
    county_names = gdf['county'].unique().tolist() if 'county' in gdf.columns else []
    mentioned_counties = [c for c in county_names if c.lower() in question.lower()]

    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition', 'health']
    stunting_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in stunting_keywords)]

    for county in mentioned_counties[:2]:
        county_df = gdf[gdf['county'] == county]
        if stunting_cols:
            for col in stunting_cols[:3]:
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

    if not extracted_data and any(w in question.lower() for w in ['stunting', 'stunt', 'malnutrition']):
        for col in stunting_cols[:2]:
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
    """Query the AI agent with the user's question and data context."""
    specific_data = extract_specific_data_for_query(gdf, question)

    system_prompt = """
You are a senior data scientist with an economics background specializing in public policy for Kenya Vision 2063.
You analyze ward-level data to provide actionable insights for policymakers.

DATASET OVERVIEW:
{data_summary}

SPECIFIC DATA FOR THIS QUERY:
{specific_data}

INSTRUCTIONS:
1. Use actual ward-level data provided above.
2. Reference specific ward names, their values, and subcounty information.
3. Provide comparative analysis and targeted policy recommendations.
4. Connect insights to Kenya Vision 2063 goals.

QUESTION: {question}
"""

    total_wards = data_summary.get("dataset_overview", {}).get("total_wards", "unknown")
    stunting_columns = data_summary.get("dataset_overview", {}).get("stunting_related_columns", [])
    specific_data_context = json.dumps(specific_data, indent=2) if specific_data else "No specific data extracted."
    data_summary_context = json.dumps({
        "dataset_overview": data_summary.get("dataset_overview", {}),
        "summary_statistics": dict(list(data_summary.get("summary_statistics", {}).items())[:5]),
        "ward_level_examples": data_summary.get("ward_level_examples", {})
    }, indent=2)

    full_prompt = system_prompt.format(
        data_summary=data_summary_context,
        specific_data=specific_data_context,
        question=question
    )

    try:
        if chat_history and len(chat_history) > 0:
            conversation_context = "\nPrevious conversation:\n"
            for msg in chat_history[-5:]:
                conversation_context += f"{msg['role']}: {msg['content']}\n"
            full_prompt = conversation_context + "\n" + full_prompt

        response = client.models.generate_content(model=model_name, contents=full_prompt)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Error querying AI agent: {str(e)}"


# FIX 2: Move cached helper functions OUTSIDE main() so cache keys are stable
@st.cache_data(ttl=300)
def get_summary_stats(_gdf, cols_tuple):
    return _gdf[list(cols_tuple)].describe().T


@st.cache_data(ttl=300)
def get_county_stats(_gdf, cols_tuple):
    return _gdf.groupby('county')[list(cols_tuple)].mean()


@st.cache_data(ttl=300)
def get_correlation(_gdf, cols_tuple):
    return _gdf[list(cols_tuple)].corr()


def main():
    st.title("📊 Kenya Ward-Level Stunting Data Explorer with AI Policy Advisor")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**App Status:** ✅ Running")

    with st.spinner("Loading ward-level data..."):
        gdf = load_geojson_from_drive()

    if gdf is None or gdf.empty:
        st.error("No data available. Please check your data source and connection.")
        st.info("1. Verify the Google Drive file ID in secrets")
        st.info("2. Ensure the file is publicly accessible")
        st.info("3. Check that the file is a valid GeoJSON")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Information**")
    st.sidebar.write(f"Columns: {len(gdf.columns)}")
    st.sidebar.write(f"Sample columns: {list(gdf.columns[:5])}...")

    numeric_cols = gdf.select_dtypes(include=['number']).columns.tolist()
    stunting_keywords = ['stunting', 'stunt', 'malnutrition', 'nutrition']
    stunting_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in stunting_keywords)]

    if stunting_cols:
        st.sidebar.success(f"✓ Found {len(stunting_cols)} stunting columns")
        st.sidebar.write("Stunting columns:", stunting_cols[:3])

    ai_client, model_name = init_gemini()

    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Wards", len(gdf))
    st.sidebar.metric("Total Counties", gdf['county'].nunique())

    if 'Ward_Codes' in numeric_cols:
        numeric_cols.remove('Ward_Codes')

    if not numeric_cols:
        st.warning("No numeric indicators found.")
        st.dataframe(gdf.head())
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map Visualization", "🤖 AI Policy Advisor", "📈 Data Analysis", "📥 Export Data"])

    with tab1:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Interactive Map")

            bounds = gdf.total_bounds
            centroid_y = (bounds[1] + bounds[3]) / 2
            centroid_x = (bounds[0] + bounds[2]) / 2

            available_indicators = stunting_cols + [c for c in numeric_cols if c not in stunting_cols] if stunting_cols else numeric_cols

            selected_indicator = st.selectbox("Select indicator to visualize:", available_indicators, key='map_indicator')

            m = create_choropleth_map(gdf, selected_indicator, centroid_y, centroid_x)

            with st.container():
                try:
                    st_folium(m, width=700, height=500, returned_objects=[], key="folium_map")
                except Exception as e:
                    st.error(f"Map rendering error: {str(e)}")

        with col2:
            st.subheader("Map Controls")
            st.metric(f"Average {selected_indicator}", f"{gdf[selected_indicator].mean():.1f}")
            st.metric(f"Highest {selected_indicator}", f"{gdf[selected_indicator].max():.1f}")
            st.metric(f"Lowest {selected_indicator}", f"{gdf[selected_indicator].min():.1f}")

            st.write("**Top 5 Wards:**")
            top_wards = gdf.nlargest(5, selected_indicator)[['ward', selected_indicator, 'county']]
            for _, row in top_wards.iterrows():
                st.write(f"- {row['ward']} ({row['county']}): {row[selected_indicator]:.1f}")

    with tab3:
        st.subheader("Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Summary Statistics**")
            # FIX 2: call the top-level cached functions; pass cols as tuple (hashable)
            summary_stats = get_summary_stats(gdf, tuple(numeric_cols))
            st.dataframe(summary_stats.style.format("{:.2f}"))

        with col2:
            st.write("**County-Level Aggregation**")
            county_stats = get_county_stats(gdf, tuple(numeric_cols))
            st.dataframe(county_stats.style.format("{:.1f}"), use_container_width=True)

        if len(numeric_cols) > 1:
            if st.checkbox("Show correlation matrix", value=False):
                st.write("**Correlation Matrix**")
                correlation = get_correlation(gdf, tuple(numeric_cols))
                st.dataframe(correlation.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))

        if stunting_cols:
            st.write("### Ward-Level Stunting Data Sample")
            sample_col = stunting_cols[0]
            sample_data = gdf[['ward', 'county', 'subcounty', sample_col]].sort_values(sample_col, ascending=False).head(10)
            st.dataframe(sample_data.style.format({sample_col: "{:.2f}"}))

    with tab4:
        st.subheader("Export Data")

        all_counties = sorted(gdf['county'].unique())
        selected_counties = st.multiselect(
            "Select counties:",
            all_counties,
            default=all_counties[:3] if len(all_counties) > 3 else all_counties
        )

        st.write("### Range Filters")
        col1, col2 = st.columns(2)

        # FIX 5: use boolean mask instead of gdf.query() to avoid column-name parsing issues
        range_masks = []
        for i, col in enumerate(numeric_cols[:2]):
            col_container = col1 if i % 2 == 0 else col2
            with col_container:
                min_val = float(gdf[col].min())
                max_val = float(gdf[col].max())
                values = st.slider(f"{col} range:", min_value=min_val, max_value=max_val,
                                   value=(min_val, max_val), key=f"export_filter_{col}")
                if values[0] > min_val or values[1] < max_val:
                    range_masks.append((gdf[col] >= values[0]) & (gdf[col] <= values[1]))

        filtered_gdf = gdf.copy()
        if selected_counties:
            filtered_gdf = filtered_gdf[filtered_gdf['county'].isin(selected_counties)]
        for mask in range_masks:
            filtered_gdf = filtered_gdf[mask.loc[filtered_gdf.index]]

        st.write(f"**Filtered Results:** {len(filtered_gdf)} of {len(gdf)} wards")

        non_geo_columns = [c for c in gdf.columns if c != 'geometry']
        export_columns = st.multiselect(
            "Select columns to export:",
            non_geo_columns,
            default=['ward', 'county', 'subcounty'] + numeric_cols[:3]
        )

        if export_columns:
            export_df = filtered_gdf[export_columns]  # geometry already excluded
            st.dataframe(export_df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # FIX 4: drop geometry (if any) before CSV export
                csv_cols = [c for c in export_columns if c != 'geometry']
                csv_data = filtered_gdf[csv_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="kenya_ward_stunting_data.csv",
                    mime="text/csv"
                )

            with col2:
                if st.button("Generate GeoJSON for download"):
                    with st.spinner("Generating GeoJSON..."):
                        # FIX 5: avoid duplicate 'geometry' in column list
                        geo_export_cols = list(dict.fromkeys(export_columns + ['geometry']))
                        available_geo_cols = [c for c in geo_export_cols if c in filtered_gdf.columns]
                        geojson_data = filtered_gdf[available_geo_cols].to_json()
                        st.download_button(
                            label="🗺️ Download as GeoJSON",
                            data=geojson_data,
                            file_name="kenya_ward_stunting_data.geojson",
                            mime="application/json"
                        )
                else:
                    st.info("Click 'Generate GeoJSON' to create download file")

    with tab2:
        st.subheader("🤖 AI Policy Advisor")
        st.markdown("""
**Ask questions about the ward-level stunting data to get insights from our AI Data Scientist.**

*Example questions:*
- Which ward in Nakuru has the highest stunting rate?
- What are the top 5 wards with highest stunting rates nationally?
- Compare stunting rates between wards in Nairobi County
- Recommend targeted interventions for high-stunting wards in Kisumu
        """)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'data_summary' not in st.session_state:
            st.session_state.data_summary = get_data_summary(gdf)

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if ai_client and model_name:
            if prompt := st.chat_input("Ask about ward-level stunting data..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing ward-level data..."):
                        response = query_ai_agent(
                            prompt,
                            st.session_state.data_summary,
                            ai_client,
                            model_name,
                            gdf,
                            st.session_state.chat_history
                        )
                        st.markdown(response)

                st.session_state.chat_history.append({"role": "assistant", "content": response})

                if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
        else:
            st.warning("⚠️ AI features are currently disabled.")
            st.info("Add your Gemini API key to `.streamlit/secrets.toml`:")
            st.code("GEMINI_API_KEY = 'your-api-key-here'")

    st.sidebar.divider()
    st.sidebar.write("### About the Dataset")
    st.sidebar.write("""
This dataset contains **ward-level stunting rates** across Kenya.

**Key Features:**
- Ward-level stunting data
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
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
