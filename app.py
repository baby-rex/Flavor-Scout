import os
from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from data_ingestion import fetch_data, DEMO_MODE
from ai_analysis import filter_noise, extract_trends, generate_recommendations

# ============================================================
# FLAVOR SCOUT: Production-Grade Trend Analytics for HealthKart
# ============================================================
# 
# ARCHITECTURE:
# 1. Data Ingestion: Multi-source (Twitter/Reddit/Mock) with graceful fallback
#    - If no credentials: falls back to mock data automatically
#    - All sources optional; app remains functional with ANY available source
# 
# 2. AI Analysis: Deterministic ranking + optional LLM explanation
#    - Trends ranked by frequency × sentiment (100% reproducible)
#    - LLM only explains WHY (not WHAT) with brand-specific tone
#    - All LLM outputs validated against source texts (zero hallucinations)
# 
# 3. UI: Streamlit dashboard with aggressive caching (< 10s load)
#    - Sidebar controls (query, brand, limit)
#    - Signal snapshot (metrics overview)
#    - Trend wall (wordcloud + chart)
#    - Decision engine (selected/rejected + golden candidate)
#
# SAFETY PRINCIPLES:
# - No hard API dependencies (Twitter/Reddit optional)
# - Transparent reasoning (every recommendation has confidence level)
# - Verified recommendations (all flavors traced back to source)
# - Conservative thresholds (low signals produce CAUTION messages)
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

st.set_page_config(page_title="Flavor Scout", page_icon="🧭", layout="wide")

# ---- Header ----
st.title("🧭 Flavor Scout")
st.markdown("**AI-powered consumer trend analytics for supplement innovation**")
st.caption("Analyze social conversations and discover data-backed flavor opportunities for HealthKart brands.")
if DEMO_MODE:
    st.caption("Demo Mode: Using 119 curated Reddit-style flavor comments (no live API).")
st.divider()

DEFAULT_QUERY = "(flavor) MuscleBlaze"
BRANDS = ["MuscleBlaze", "HK Vitals", "TrueBasics"]


# -------------------------
# Cached data helpers
# -------------------------
# Why caching? Streamlit reruns entire script on every interaction.
# Caching prevents redundant API calls and ensures < 10s load times.
# @st.cache_data persists across sessions, @st.cache_resource for stateful objects.
@st.cache_data(show_spinner=False)
def cached_fetch(query: str, limit: int = 200) -> pd.DataFrame:
    return fetch_data(query=query, limit=limit)


@st.cache_data(show_spinner=False)
def cached_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Filter once per DataFrame - query-aware filtering happens in extraction
    return filter_noise(df)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda x: id(x)})
def cached_trends(df: pd.DataFrame, query: str) -> Dict[str, Dict[str, Any]]:
    # Cache key uses DataFrame id + query string
    return extract_trends(df, query)


@st.cache_data(show_spinner=False, hash_funcs={dict: lambda x: id(x)})
def cached_recos(trends: Dict[str, Dict[str, Any]], brand: str) -> Dict[str, Any]:
    # Cache key uses trends dict id + brand
    return generate_recommendations(trends, brand)


# -------------------------
# Sidebar controls
# -------------------------
# ---- Sidebar Controls ----
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.caption("Configure your trend analysis")
    
    query = st.text_input(
        "Search Query",
        value=DEFAULT_QUERY,
        help="Keywords to focus on (e.g., flavors, ingredients, brand names)",
    )
    
    brand = st.selectbox(
        "Target Brand",
        options=BRANDS,
        index=0,
        help="Recommendations will be tailored to this brand's tone"
    )
    
    limit = st.slider(
        "Data Volume",
        min_value=50,
        max_value=400,
        value=200,
        step=50,
        help="Maximum social media posts to analyze"
    )
    
    st.divider()
    analyze = st.button("🔍 Analyze Trends", type="primary", use_container_width=True)


# -------------------------
# Visualization helpers
# -------------------------
def render_wordcloud(trends: Dict[str, Dict[str, Any]], top_n: int = 15):
    """
    Render bar chart of trending flavors.
    
    Why bar chart? Clear visualization of frequency with exact values.
    Easier to compare flavor mentions and identify trends at a glance.
    """
    freqs_full = {
        k: max(v.get("freq", 1), 1)
        for k, v in trends.items()
        if k and isinstance(v, dict)
    }

    freqs_items = sorted(freqs_full.items(), key=lambda x: x[1], reverse=True)[:top_n]
    freqs = dict(freqs_items)

    if not freqs:
        st.info("No trend terms available to render.")
        return

    # Render bar chart as primary visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    labels, values = zip(*freqs_items)
    bars = ax.bar(labels, values, color="#ff9800", edgecolor="#333", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Flavors", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mentions", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {len(freqs)} Trending Flavors", fontsize=14, fontweight='bold')
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig)


# -------------------------
# Main app flow
# -------------------------
# Why this structure? Each stage represents a deliberate decision point:
# 1. Fetch: Ingest from all available sources (Twitter/Reddit/Mock)
# 2. Filter: Remove noise using deterministic rules (keyword matching)
# 3. Extract: Find trends via LLM with validation layer (zero hallucinations)
# 4. Recommend: Rank by frequency × sentiment and explain with brand tone
# Each stage is cached to prevent redundant API calls and ensure < 10s load.
if analyze:
    with st.spinner("Listening to chatter and analyzing flavors..."):
        df_raw = cached_fetch(query=query or DEFAULT_QUERY, limit=limit)

        if df_raw.empty:
            st.warning("No data found for this query. Try broadening it.")

        else:
            df_filtered = cached_filter(df_raw)

            if df_filtered.empty:
                st.warning("Data fetched, but no flavor-relevant posts after filtering.")

            else:
                trends = cached_trends(df_filtered, query)
                recos = cached_recos(trends, brand)

                sentiments = [
                    v.get("sentiment", 0)
                    for v in trends.values()
                    if isinstance(v, dict)
                ]
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

                # ---- Metrics Row ----
                st.markdown("")
                st.subheader("📊 Consumer Signal Overview")
                st.caption("Key metrics from your social media analysis")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Posts Collected", len(df_raw), help="Total posts fetched from social sources")
                m2.metric("Relevant Posts", len(df_filtered), delta=f"{len(df_filtered)}/{len(df_raw)}", help="After noise filtering")
                m3.metric("Unique Flavors", len(trends), help="Distinct flavor trends detected")
                m4.metric("Avg. Sentiment", f"{avg_sentiment:.0%}", delta="Positive" if avg_sentiment > 0.5 else "Mixed", help="Overall consumer sentiment")
                
                st.divider()

                # ---- Flavor Opportunity Map ----
                st.subheader("🗺️ Flavor Opportunity Map")
                st.caption("Larger text = higher consumer interest. Visual snapshot of trending flavors.")
                render_wordcloud(trends, top_n=15)
                st.divider()

                # ---- Decision Engine ----
                st.subheader("🎯 Recommendation Engine")
                st.caption("AI-validated flavor recommendations for your brand, ranked by consumer demand and sentiment.")
                
                selected = recos.get("selected", []) or []
                rejected = recos.get("rejected", []) or []

                cols = st.columns([1, 1], gap="large")

                with cols[0]:
                    st.markdown("### ✅ Strong Candidates")
                    if selected:
                        for s in selected:
                            with st.container():
                                st.markdown(f"**{s.get('flavor', '')}**")
                                st.markdown(f"<div style='color: #666; padding-left: 1rem; font-style: italic;'>{s.get('why', '')}</div>", unsafe_allow_html=True)
                                st.markdown("")
                    else:
                        st.info("💡 No strong candidates identified. Consider broadening your query.")

                with cols[1]:
                    st.markdown("### ⚠️ Not Recommended")
                    if rejected:
                        for r in rejected:
                            with st.container():
                                st.markdown(f"**{r.get('flavor', '')}**")
                                st.markdown(f"<div style='color: #888; padding-left: 1rem; font-style: italic;'>{r.get('why', '')}</div>", unsafe_allow_html=True)
                                st.markdown("")
                    else:
                        st.info("All detected flavors passed quality thresholds.")

                # ---- Golden Candidate Highlight ----
                st.divider()
                golden = recos.get("golden", {"flavor": "", "why": ""})
                
                if golden.get("flavor"):
                    st.markdown("")
                    st.markdown("### 🏆 Top Recommendation")
                    st.success(
                        f"**{golden.get('flavor')}**\n\n"
                        f"{golden.get('why')}\n\n"
                        f"*Highest-ranked by consumer frequency and positive sentiment.*"
                    )
                else:
                    st.info("💡 No clear winner emerged. Consider refining your search query or analyzing more data.")

                st.markdown("")
                with st.expander("📄 View Raw Data Samples (Top 20)"):
                    st.caption("Sample of filtered social media posts used in this analysis")
                    st.dataframe(df_filtered.head(20), use_container_width=True)

else:
    # ---- Welcome State ----
    st.markdown("")
    st.info(
        "👈 **Get started:** Configure your analysis settings in the sidebar, then click **Analyze Trends**.\n\n"
        "**Example queries to try:**\n"
        "- `kesar pista indian flavor whey` — Traditional Indian flavors\n"
        "- `dark cocoa less sweet whey` — Premium flavor profiles\n"
        "- `blueberry electrolyte drink` — Emerging fruit trends"
    )
    
    st.markdown("")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📡 Multi-Source Data**")
        st.caption("Aggregates trends from Twitter, Reddit, and curated samples")
    with col2:
        st.markdown("**🤖 AI-Validated**")
        st.caption("Every recommendation verified against source data — zero hallucinations")
    with col3:
        st.markdown("**🎯 Brand-Aware**")
        st.caption("Recommendations tailored to your selected brand's tone and positioning")
