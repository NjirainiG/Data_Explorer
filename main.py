from google import genai
import streamlit as st
import requests
import geopandas as gpd
import folium
import pandas as pd
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CHAT_HISTORY = 20
SIMPLIFICATION_TOLERANCE = 0.001

st.set_page_config(
    page_title="Kenya 2063 Ward Level Data Explorer",
    page_icon="📍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Colour helpers  (module-level so they are always picklable / importable)
# ---------------------------------------------------------------------------

def get_color(value, min_val, max_val):
    """Return a hex colour from the YlOrRd ramp for *value*."""
    if pd.isna(value):
        return "#808080"
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    rgba = cm.YlOrRd(norm(value))
    return mcolors.to_hex(rgba)


# FIX: style_function must be at MODULE level — nested (local) functions
# cannot be pickled, which causes the st.cache_data AttributeError.
# We pass min/max via a factory that returns a plain closure-free callable
# by baking the values into default arguments (default args ARE picklable).
def _make_style_fn(indicator: str, min_val: float, max_val: float):
    """Return a Folium style function with baked-in parameters."""

    def style_function(
        feature,
        _ind=indicator,
        _min=min_val,
        _max=max_val,
    ):
        try:
            value = feature.get("properties", {}).get(_ind)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return {
                    "fillColor": "#808080",
                    "color": "#666666",
                    "weight": 0.3,
                    "fillOpacity": 0.5,
                }
            return {
                "fillColor": get_color(value, _min, _max),
                "color": "#666666",
                "weight": 0.3,
                "fillOpacity": 0.7,
            }
        except Exception:
            return {
                "fillColor": "#808080",
                "color": "#666666",
                "weight": 0.3,
                "fillOpacity": 0.5,
            }

    return style_function


# ---------------------------------------------------------------------------
# Gemini initialisation
# ---------------------------------------------------------------------------

@st.cache_resource
def init_gemini():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.warning("GEMINI_API_KEY not found in secrets. AI features disabled.")
            return None, None

        client = genai.Client(api_key=api_key)

        for model_name in [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
        ]:
            try:
                resp = client.models.generate_content(model=model_name, contents="Hello")
                if resp and hasattr(resp, "text"):
                    st.sidebar.success(f"✓ Using model: {model_name}")
                    return client, model_name
            except Exception:
                continue

        st.error("No compatible Gemini model found.")
        return None, None

    except Exception as e:
        st.error(f"Failed to initialise Gemini API: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Google Drive download
# ---------------------------------------------------------------------------

def download_file_from_google_drive(file_id: str):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True, timeout=30)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(
                URL, params={"id": file_id, "confirm": value}, stream=True, timeout=30
            )
            break
    return response


@st.cache_data(ttl=3600, show_spinner="Loading ward data…")
def load_geojson_from_drive() -> gpd.GeoDataFrame:
    try:
        file_id = st.secrets.get("GOOGLE_DRIVE_GEOJSON_FILE_ID")
        if not file_id:
            st.error("Google Drive file ID not configured in secrets.")
            return gpd.GeoDataFrame()

        response = download_file_from_google_drive(file_id)
        response.raise_for_status()

        content = response.content
        if not content:
            st.error("Received empty response from Google Drive.")
            return gpd.GeoDataFrame()

        try:
            text = content.decode("utf-8")
            json.loads(text)  # validate JSON
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            st.error(f"Invalid GeoJSON: {e}")
            return gpd.GeoDataFrame()

        try:
            gdf = gpd.read_file(io.StringIO(text))
        except Exception as e:
            st.error(f"Error parsing GeoJSON: {e}")
            return gpd.GeoDataFrame()

        if gdf.empty:
            st.warning("Loaded GeoDataFrame is empty.")
            return gdf

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)

        gdf["geometry"] = gdf["geometry"].make_valid()
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFICATION_TOLERANCE)

        for col in gdf.select_dtypes(include=["number"]).columns:
            gdf[col] = pd.to_numeric(gdf[col], downcast="float")

        st.sidebar.success(f"✓ Loaded {len(gdf)} wards")
        return gdf

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return gpd.GeoDataFrame()
    except Exception as e:
        import traceback
        st.error(f"Error loading data: {e}\n{traceback.format_exc()}")
        return gpd.GeoDataFrame()


# ---------------------------------------------------------------------------
# Map creation
# FIX: @st.cache_data REMOVED — Folium Map objects are not picklable.
#      The style function is now created via _make_style_fn (module-level factory)
#      so there is no unpicklable local closure anywhere.
# ---------------------------------------------------------------------------

def create_choropleth_map(
    gdf: gpd.GeoDataFrame,
    indicator: str,
    centroid_y: float,
    centroid_x: float,
) -> folium.Map:
    try:
        m = folium.Map(
            location=[centroid_y, centroid_x],
            zoom_start=6,
            tiles="cartodbpositron",
            control_scale=True,
            prefer_canvas=True,
        )

        valid_values = gdf[indicator].dropna()
        if valid_values.empty:
            st.warning(f"No valid values for {indicator}.")
            return m

        min_val = float(valid_values.min())
        max_val = float(valid_values.max())

        geojson_data = json.loads(gdf.to_json())

        # Module-level factory → no unpicklable local closure
        style_fn = _make_style_fn(indicator, min_val, max_val)

        tooltip_fields, tooltip_aliases = [], []
        for field, alias in [
            ("ward", "Ward:"),
            (indicator, f"{indicator}:"),
            ("county", "County:"),
        ]:
            if field in gdf.columns:
                tooltip_fields.append(field)
                tooltip_aliases.append(alias)

        folium.GeoJson(
            geojson_data,
            name="choropleth",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                style="font-size: 11px;",
            ),
        ).add_to(m)

        colormap = LinearColormap(
            colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
            vmin=min_val,
            vmax=max_val,
            caption=indicator,
        )
        colormap.add_to(m)

        return m

    except Exception as e:
        import traceback
        st.error(f"Error creating map: {e}\n{traceback.format_exc()}")
        return folium.Map(location=[centroid_y, centroid_x], zoom_start=6, tiles="cartodbpositron")


# ---------------------------------------------------------------------------
# Data summary helpers
# ---------------------------------------------------------------------------

def get_data_summary(gdf: gpd.GeoDataFrame) -> dict:
    numeric_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
    if "Ward_Codes" in numeric_cols:
        numeric_cols.remove("Ward_Codes")

    stunting_kw = ["stunting", "stunt", "malnutrition", "nutrition", "health", "wasting", "underweight"]
    stunting_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in stunting_kw)]

    county_ward_data: dict = {}
    if "county" in gdf.columns:
        for county in gdf["county"].unique():
            cdf = gdf[gdf["county"] == county]
            if not stunting_cols:
                continue
            county_ward_data[county] = {}
            for col in stunting_cols[:3]:
                if col not in cdf.columns:
                    continue
                valid = cdf[col].dropna()
                if valid.empty:
                    continue
                max_idx = cdf[col].idxmax(skipna=True)
                min_idx = cdf[col].idxmin(skipna=True)
                mxr, mnr = cdf.loc[max_idx], cdf.loc[min_idx]
                county_ward_data[county][col] = {
                    "highest_ward": {
                        "ward": mxr.get("ward", "N/A"),
                        "value": float(mxr[col]),
                        "subcounty": mxr.get("subcounty", "N/A"),
                    },
                    "lowest_ward": {
                        "ward": mnr.get("ward", "N/A"),
                        "value": float(mnr[col]),
                        "subcounty": mnr.get("subcounty", "N/A"),
                    },
                    "county_average": float(cdf[col].mean()),
                    "ward_count": len(cdf),
                }

    summary: dict = {
        "dataset_overview": {
            "data_granularity": "WARD-LEVEL",
            "total_wards": len(gdf),
            "total_counties": gdf["county"].nunique() if "county" in gdf.columns else 0,
            "total_subcounties": gdf["subcounty"].nunique() if "subcounty" in gdf.columns else 0,
            "columns": gdf.columns.tolist(),
            "numeric_columns": numeric_cols,
            "stunting_related_columns": stunting_cols,
            "has_ward_level_stunting_data": bool(stunting_cols),
            "stunting_columns_found": stunting_cols,
        },
        "summary_statistics": {},
        "ward_level_examples": {"sample_ward_data": {}},
        "county_ward_analysis": county_ward_data,
        "top_bottom_wards": {},
    }

    for col in numeric_cols:
        summary["summary_statistics"][col] = {
            "mean": float(gdf[col].mean()),
            "median": float(gdf[col].median()),
            "min": float(gdf[col].min()),
            "max": float(gdf[col].max()),
            "std": float(gdf[col].std()),
        }
        cols_present = [c for c in ["ward", "county", "subcounty", col] if c in gdf.columns]
        summary["top_bottom_wards"][col] = {
            "top_5_wards": gdf.nlargest(5, col)[cols_present].to_dict("records"),
            "bottom_5_wards": gdf.nsmallest(5, col)[cols_present].to_dict("records"),
        }

    return summary


def extract_specific_data_for_query(gdf: gpd.GeoDataFrame, question: str) -> dict:
    extracted: dict = {}
    county_names = gdf["county"].unique().tolist() if "county" in gdf.columns else []
    mentioned = [c for c in county_names if c.lower() in question.lower()]

    numeric_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
    stunting_kw = ["stunting", "stunt", "malnutrition", "nutrition", "health"]
    stunting_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in stunting_kw)]

    for county in mentioned[:2]:
        cdf = gdf[gdf["county"] == county]
        for col in stunting_cols[:3]:
            top = cdf.nlargest(5, col)[["ward", col, "subcounty"]]
            bot = cdf.nsmallest(5, col)[["ward", col, "subcounty"]]
            extracted[f"{county}_{col}"] = {
                "county": county,
                "indicator": col,
                "top_wards": top.to_dict("records"),
                "bottom_wards": bot.to_dict("records"),
                "county_average": float(cdf[col].mean()),
                "ward_count": len(cdf),
            }

    if not extracted and any(w in question.lower() for w in ["stunting", "stunt", "malnutrition"]):
        for col in stunting_cols[:2]:
            cols_present = [c for c in ["ward", "county", "subcounty", col] if c in gdf.columns]
            extracted[f"national_{col}"] = {
                "indicator": col,
                "top_wards_national": gdf.nlargest(10, col)[cols_present].to_dict("records"),
                "bottom_wards_national": gdf.nsmallest(10, col)[cols_present].to_dict("records"),
                "national_average": float(gdf[col].mean()),
            }

    return extracted


def query_ai_agent(
    question: str,
    data_summary: dict,
    client,
    model_name: str,
    gdf: gpd.GeoDataFrame,
    chat_history: list | None = None,
) -> str:
    specific_data = extract_specific_data_for_query(gdf, question)

    system_prompt = """
You are a senior data scientist specialising in public policy for Kenya Vision 2063.
Analyse ward-level stunting data and give actionable insights for policymakers.

DATASET OVERVIEW:
{data_summary}

SPECIFIC DATA FOR THIS QUERY:
{specific_data}

INSTRUCTIONS:
1. Reference specific ward names, exact values, and subcounty information.
2. Provide comparative analysis and targeted policy recommendations.
3. Connect insights to Kenya Vision 2063 goals.

QUESTION: {question}
""".format(
        data_summary=json.dumps(
            {
                "dataset_overview": data_summary.get("dataset_overview", {}),
                "summary_statistics": dict(list(data_summary.get("summary_statistics", {}).items())[:5]),
                "ward_level_examples": data_summary.get("ward_level_examples", {}),
            },
            indent=2,
        ),
        specific_data=json.dumps(specific_data, indent=2) if specific_data else "No specific data extracted.",
        question=question,
    )

    if chat_history:
        history_text = "\nPrevious conversation:\n" + "".join(
            f"{m['role']}: {m['content']}\n" for m in chat_history[-5:]
        )
        system_prompt = history_text + "\n" + system_prompt

    try:
        resp = client.models.generate_content(model=model_name, contents=system_prompt)
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        return f"Error querying AI agent: {e}"


# ---------------------------------------------------------------------------
# Cached analytics helpers  (must be at module level for stable cache keys)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_summary_stats(_gdf, cols_tuple):
    return _gdf[list(cols_tuple)].describe().T


@st.cache_data(ttl=300)
def get_county_stats(_gdf, cols_tuple):
    return _gdf.groupby("county")[list(cols_tuple)].mean()


@st.cache_data(ttl=300)
def get_correlation(_gdf, cols_tuple):
    return _gdf[list(cols_tuple)].corr()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("📊 Kenya Ward-Level Stunting Data Explorer with AI Policy Advisor")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**App Status:** ✅ Running")

    with st.spinner("Loading ward-level data…"):
        gdf = load_geojson_from_drive()

    if gdf is None or gdf.empty:
        st.error("No data available. Check your data source and connection.")
        st.info("1. Verify the Google Drive file ID in secrets")
        st.info("2. Ensure the file is publicly accessible")
        st.info("3. Check that the file is valid GeoJSON")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Information**")
    st.sidebar.write(f"Columns: {len(gdf.columns)}")
    st.sidebar.write(f"Sample columns: {list(gdf.columns[:5])}…")

    numeric_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
    if "Ward_Codes" in numeric_cols:
        numeric_cols.remove("Ward_Codes")

    stunting_kw = ["stunting", "stunt", "malnutrition", "nutrition"]
    stunting_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in stunting_kw)]

    if stunting_cols:
        st.sidebar.success(f"✓ Found {len(stunting_cols)} stunting columns")
        st.sidebar.write("Stunting columns:", stunting_cols[:3])

    ai_client, model_name = init_gemini()

    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Wards", len(gdf))
    if "county" in gdf.columns:
        st.sidebar.metric("Total Counties", gdf["county"].nunique())

    if not numeric_cols:
        st.warning("No numeric indicators found.")
        st.dataframe(gdf.head())
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Map Visualization", "🤖 AI Policy Advisor", "📈 Data Analysis", "📥 Export Data"]
    )

    # ------------------------------------------------------------------ Tab 1
    with tab1:
        col_map, col_ctrl = st.columns([3, 1])

        with col_map:
            st.subheader("Interactive Map")

            bounds = gdf.total_bounds
            centroid_y = (bounds[1] + bounds[3]) / 2
            centroid_x = (bounds[0] + bounds[2]) / 2

            available_indicators = (
                stunting_cols + [c for c in numeric_cols if c not in stunting_cols]
                if stunting_cols
                else numeric_cols
            )

            selected_indicator = st.selectbox(
                "Select indicator to visualise:", available_indicators, key="map_indicator"
            )

            # create_choropleth_map is NOT cached (Folium maps cannot be pickled)
            m = create_choropleth_map(gdf, selected_indicator, centroid_y, centroid_x)

            try:
                st_folium(
                    m,
                    width=700,
                    height=500,
                    returned_objects=[],
                    key="folium_map",
                )
            except Exception as e:
                st.error(f"Map rendering error: {e}")

        with col_ctrl:
            st.subheader("Statistics")
            st.metric(f"Average {selected_indicator}", f"{gdf[selected_indicator].mean():.1f}")
            st.metric(f"Highest {selected_indicator}", f"{gdf[selected_indicator].max():.1f}")
            st.metric(f"Lowest {selected_indicator}", f"{gdf[selected_indicator].min():.1f}")

            st.write("**Top 5 Wards:**")
            top5_cols = [c for c in ["ward", selected_indicator, "county"] if c in gdf.columns]
            for _, row in gdf.nlargest(5, selected_indicator)[top5_cols].iterrows():
                county_label = f" ({row['county']})" if "county" in row else ""
                st.write(f"- {row.get('ward', '?')}{county_label}: {row[selected_indicator]:.1f}")

    # ------------------------------------------------------------------ Tab 3
    with tab3:
        st.subheader("Data Analysis")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Summary Statistics**")
            st.dataframe(get_summary_stats(gdf, tuple(numeric_cols)).style.format("{:.2f}"))

        with c2:
            st.write("**County-Level Aggregation**")
            if "county" in gdf.columns:
                st.dataframe(
                    get_county_stats(gdf, tuple(numeric_cols)).style.format("{:.1f}"),
                    use_container_width=True,
                )

        if len(numeric_cols) > 1 and st.checkbox("Show correlation matrix", value=False):
            st.write("**Correlation Matrix**")
            st.dataframe(
                get_correlation(gdf, tuple(numeric_cols)).style.background_gradient(
                    cmap="RdBu", vmin=-1, vmax=1
                )
            )

        if stunting_cols:
            st.write("### Ward-Level Stunting Data Sample")
            scol = stunting_cols[0]
            sample_cols = [c for c in ["ward", "county", "subcounty", scol] if c in gdf.columns]
            st.dataframe(
                gdf[sample_cols].sort_values(scol, ascending=False).head(10).style.format({scol: "{:.2f}"})
            )

    # ------------------------------------------------------------------ Tab 4
    with tab4:
        st.subheader("Export Data")

        if "county" in gdf.columns:
            all_counties = sorted(gdf["county"].unique())
            selected_counties = st.multiselect(
                "Select counties:",
                all_counties,
                default=all_counties[:3] if len(all_counties) > 3 else all_counties,
            )
        else:
            selected_counties = []

        st.write("### Range Filters")
        c1, c2 = st.columns(2)
        range_masks = []
        for i, col in enumerate(numeric_cols[:2]):
            with (c1 if i % 2 == 0 else c2):
                lo, hi = float(gdf[col].min()), float(gdf[col].max())
                sel = st.slider(f"{col} range:", min_value=lo, max_value=hi,
                                value=(lo, hi), key=f"export_filter_{col}")
                if sel[0] > lo or sel[1] < hi:
                    range_masks.append((gdf[col] >= sel[0]) & (gdf[col] <= sel[1]))

        filtered = gdf.copy()
        if selected_counties:
            filtered = filtered[filtered["county"].isin(selected_counties)]
        for mask in range_masks:
            filtered = filtered[mask.loc[filtered.index]]

        st.write(f"**Filtered Results:** {len(filtered)} of {len(gdf)} wards")

        non_geo_cols = [c for c in gdf.columns if c != "geometry"]
        default_export = [c for c in (["ward", "county", "subcounty"] + numeric_cols[:3]) if c in non_geo_cols]
        export_columns = st.multiselect("Select columns to export:", non_geo_cols, default=default_export)

        if export_columns:
            st.dataframe(filtered[export_columns], use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="📥 Download as CSV",
                    data=filtered[export_columns].to_csv(index=False),
                    file_name="kenya_ward_stunting_data.csv",
                    mime="text/csv",
                )
            with dl2:
                if st.button("Generate GeoJSON for download"):
                    with st.spinner("Generating GeoJSON…"):
                        geo_cols = list(dict.fromkeys(export_columns + ["geometry"]))
                        geo_cols = [c for c in geo_cols if c in filtered.columns]
                        st.download_button(
                            label="🗺️ Download as GeoJSON",
                            data=filtered[geo_cols].to_json(),
                            file_name="kenya_ward_stunting_data.geojson",
                            mime="application/json",
                        )
                else:
                    st.info("Click 'Generate GeoJSON' to create download file")

    # ------------------------------------------------------------------ Tab 2
    with tab2:
        st.subheader("🤖 AI Policy Advisor")
        st.markdown("""
**Ask questions about ward-level stunting data for AI-powered insights.**

*Example questions:*
- Which ward in Nakuru has the highest stunting rate?
- Top 5 wards with highest stunting rates nationally?
- Compare stunting between wards in Nairobi County
- Recommend interventions for high-stunting wards in Kisumu
        """)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "data_summary" not in st.session_state:
            st.session_state.data_summary = get_data_summary(gdf)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if ai_client and model_name:
            if prompt := st.chat_input("Ask about ward-level stunting data…"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analysing ward-level data…"):
                        answer = query_ai_agent(
                            prompt,
                            st.session_state.data_summary,
                            ai_client,
                            model_name,
                            gdf,
                            st.session_state.chat_history,
                        )
                        st.markdown(answer)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
        else:
            st.warning("⚠️ AI features are disabled.")
            st.info("Add your Gemini API key to `.streamlit/secrets.toml`:")
            st.code("GEMINI_API_KEY = 'your-api-key-here'")

    st.sidebar.divider()
    st.sidebar.write("### About the Dataset")
    st.sidebar.write("""
Ward-level stunting rates across Kenya.

**Key Features:**
- Most granular admin unit (ward-level)
- County & subcounty context
- Geographic boundaries for mapping
- Multiple stunting/nutrition indicators

**Data Source:** Google Drive  
**AI:** Powered by Google Gemini
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Application error: {e}")
        st.error(traceback.format_exc())
