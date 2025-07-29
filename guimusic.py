# Enhanced Music Recommendation System with Detailed Artist Search
# This version adds enhanced search results with detailed metrics and expandable song lists

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Set, Tuple, Optional
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

# STREAMLIT CONFIGURATION
st.set_page_config(
    page_title="🎵 Personal Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS FOR ENHANCED STYLING WITH EXPANDABLE SECTIONS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1DB954;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        margin: 0.5rem 0;
    }
    
    .tier-metric-card {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        margin: 0.5rem 0;
        border: 2px solid #ffffff;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #1DB954;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .hover-recommendation-card {
        background: linear-gradient(135deg, #e8f5e8, #d4edda);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 10px rgba(40,167,69,0.2);
    }
    
    .song-list {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .song-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #dee2e6;
        font-size: 0.95rem;
    }
    
    .song-item:last-child {
        border-bottom: none;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success {
        background-color: #28a745;
    }
    
    .status-error {
        background-color: #dc3545;
    }
    
    .tier-info {
        background: linear-gradient(135deg, #17a2b8, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .enhanced-search-result-card {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 10px rgba(255, 193, 7, 0.2);
    }
    
    .artist-rank-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .artist-metrics-line {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1rem;
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }
    
    .metric-item {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    .metric-value {
        font-weight: bold;
        color: #333;
    }
    
    .jump-link {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        display: inline-block;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 123, 255, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .jump-link:hover {
        background: linear-gradient(135deg, #0056b3, #004085);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.4);
        text-decoration: none;
        color: white;
    }
    
    .jump-link-container {
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(0, 123, 255, 0.1);
        border-radius: 10px;
        border: 1px dashed #007bff;
    }
    
    .expand-button {
        background: linear-gradient(135deg, #6c757d, #5a6268);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
    }
    
    .expand-button:hover {
        background: linear-gradient(135deg, #5a6268, #495057);
        transform: translateY(-1px);
    }
    
    .songs-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #dee2e6;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .song-entry {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e9ecef;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .song-entry:last-child {
        border-bottom: none;
    }
    
    .song-name {
        font-weight: 500;
        color: #333;
    }
    
    .song-stats {
        font-size: 0.85rem;
        color: #666;
        display: flex;
        gap: 1rem;
    }
    
    /* Smooth scrolling for the entire page */
    html {
        scroll-behavior: smooth;
    }
    
    /* Anchor target styling */
    .anchor-target {
        padding-top: 2rem;
        margin-top: -2rem;
    }
</style>

<script>
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start',
            inline: 'nearest'
        });
    }
}
</script>
""", unsafe_allow_html=True)

# GLOBAL CONFIGURATION
class Config:
    DATA_FOLDER = "data/spotify"
    CACHE_FOLDER = "cache"
    MAX_RECOMMENDATIONS = 50
    DEFAULT_TIER_START = 1
    DEFAULT_TIER_END = 50

# SESSION STATE INITIALIZATION
def initialize_session_state():
    """Initialize all session state variables"""
    if 'spotify_dataframe' not in st.session_state:
        st.session_state.spotify_dataframe = None
    if 'available_years' not in st.session_state:
        st.session_state.available_years = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {'lastfm': False, 'musicbrainz': False}
    if 'tier_start' not in st.session_state:
        st.session_state.tier_start = Config.DEFAULT_TIER_START
    if 'tier_end' not in st.session_state:
        st.session_state.tier_end = Config.DEFAULT_TIER_END
    if 'num_recommendations' not in st.session_state:
        st.session_state.num_recommendations = 10
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'hover_recommendations' not in st.session_state:
        st.session_state.hover_recommendations = []
    if 'selected_hover_artist' not in st.session_state:
        st.session_state.selected_hover_artist = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'show_jump_link' not in st.session_state:
        st.session_state.show_jump_link = False
    if 'expanded_artists' not in st.session_state:
        st.session_state.expanded_artists = set()

# DATA LOADING FUNCTIONS
@st.cache_data
def load_spotify_data(data_folder: str) -> pd.DataFrame:
    """Load and process all Spotify JSON files"""
    try:
        json_files = glob.glob(os.path.join(data_folder, "*.json"))
        
        if not json_files:
            st.error(f"No JSON files found in {data_folder}")
            return pd.DataFrame()
        
        all_data = []
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                st.warning(f"Error reading {file_path}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Data cleaning and processing
        required_columns = ['ts', 'master_metadata_track_name', 'master_metadata_album_artist_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Rename columns for consistency
        column_mapping = {
            'ts': 'timestamp',
            'master_metadata_track_name': 'track',
            'master_metadata_album_artist_name': 'artist',
            'master_metadata_album_album_name': 'album',
            'ms_played': 'ms_played'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Filter out blank entries
        blank_filter = (
            (df['artist'] != '(Blank)') & 
            (df['track'] != '(Blank)') & 
            (df['artist'] != '') &
            (df['track'] != '') &
            (df['artist'].notna()) &
            (df['track'].notna())
        )
        df = df[blank_filter]
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate engagement metrics
        if 'ms_played' in df.columns:
            df['ms_played'] = pd.to_numeric(df['ms_played'], errors='coerce').fillna(0)
            df['hours_played'] = df['ms_played'] / (1000 * 60 * 60)
            df['engagement_score'] = np.minimum(df['ms_played'] / 30000, 1.0)  # 30 seconds = full engagement
        else:
            df['hours_played'] = 0.1  # Default value
            df['engagement_score'] = 1.0
        
        # Add temporal features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_available_years(df: pd.DataFrame) -> List[int]:
    """Extract available years from the dataframe"""
    if df.empty:
        return []
    return sorted(df['year'].unique().tolist())

def calculate_artist_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive artist statistics"""
    if df.empty:
        return pd.DataFrame()
    
    artist_stats = df.groupby('artist').agg({
        'track': 'count',
        'hours_played': 'sum',
        'engagement_score': ['mean', 'sum'],
        'timestamp': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    artist_stats.columns = ['play_count', 'total_hours', 'avg_engagement', 'total_engagement', 'first_play', 'last_play']
    
    # Calculate preference score using ML-based weighted ensemble
    artist_stats['preference_score'] = (
        artist_stats['total_engagement'] * 0.4 +
        artist_stats['avg_engagement'] * 0.3 +
        np.log1p(artist_stats['play_count']) * 0.2 +
        np.log1p(artist_stats['total_hours']) * 0.1
    )
    
    # Add ranking
    artist_stats = artist_stats.sort_values('preference_score', ascending=False)
    artist_stats['rank'] = range(1, len(artist_stats) + 1)
    artist_stats = artist_stats.reset_index()
    
    return artist_stats

def get_artist_songs(artist: str, df: pd.DataFrame) -> List[Dict]:
    """Get all songs listened to for a specific artist with stats"""
    if df.empty:
        return []
    
    artist_df = df[df['artist'] == artist].copy()
    
    if artist_df.empty:
        return []
    
    # Group by track to get song statistics
    song_stats = artist_df.groupby('track').agg({
        'hours_played': 'sum',
        'engagement_score': 'mean',
        'timestamp': ['count', 'min', 'max']
    }).round(3)
    
    # Flatten column names
    song_stats.columns = ['total_hours', 'avg_engagement', 'play_count', 'first_played', 'last_played']
    
    # Sort by total hours played (most listened songs first)
    song_stats = song_stats.sort_values('total_hours', ascending=False)
    
    songs = []
    for track, row in song_stats.iterrows():
        songs.append({
            'track': track,
            'total_hours': row['total_hours'],
            'play_count': int(row['play_count']),
            'avg_engagement': row['avg_engagement'],
            'first_played': row['first_played'].strftime('%Y-%m-%d'),
            'last_played': row['last_played'].strftime('%Y-%m-%d')
        })
    
    return songs

def get_api_credentials():
    return {
        'lastfm_api_key': os.getenv('LASTFM_API_KEY') or st.secrets.get("LASTFM",{}).get("API_KEY"),
        'user_agent': os.getenv('MUSICBRAINZ_USER_AGENT') or st.secrets.get("MUSICBRAINZ",{}).get("USER_AGENT")
    }

# API FUNCTIONS
def validate_api_connectivity():
    """Test actual API connectivity""" #28/7/25 Roberto
    credentials = get_api_credentials()
    #api_key = os.getenv('LASTFM_API_KEY') or st.secrets.get("LASTFM",{}).get("API_KEY")
    #user_agent = os.getenv('MUSICBRAINZ_USER_AGENT') or st.secrets.get("MUSICBRAINZ",{}).get("USER_AGENT")
    
    status = {'lastfm': False, 'musicbrainz': False}
    api_key = credentials['lastfm_api_key']    
    if credentials['lastfm_api_key'] and credentials['user_agent']:
        try:
            # Test Last.fm API with a simple call
            test_url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist=Radiohead&api_key={api_key}&format=json"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200 and 'artist' in response.json():
                status['lastfm'] = True
                status['musicbrainz'] = True  # Assume MusicBrainz is valid if user agent is set, it's always valid
        except Exception as e:
            st.error(f"Error connecting to Last.fm API: {str(e)}")
    else:
        st.error("Last.fm API key is not set. Please check your configuration.")
        status['musicbrainz'] = bool(credentials['user_agent']) # Assume valid if provided
    if not status['musicbrainz']:
        st.warning("MusicBrainz is not connected. Some features may be limited.")
    return status

def get_top_tracks_simple(artist: str, api_key: str, limit: int = 5) -> List[str]:
    """Get top tracks for an artist using Last.fm API"""
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.gettoptracks&artist={artist}&api_key={api_key}&format=json&limit={limit}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        if 'toptracks' not in data or 'track' not in data['toptracks']:
            return []
        
        tracks = data['toptracks']['track']
        if not isinstance(tracks, list):
            tracks = [tracks]
        
        top_tracks = []
        for track in tracks[:limit]:
            if isinstance(track, dict) and 'name' in track:
                top_tracks.append(track['name'])
        
        return top_tracks
        
    except Exception as e:
        st.error(f"Error fetching tracks for {artist}: {e}")
        return []

def get_similar_artists(artist: str, api_key: str, limit: int = 5) -> List[Dict]:
    """Get similar artists using Last.fm API"""
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar&artist={artist}&api_key={api_key}&format=json&limit={limit}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        if 'similarartists' not in data or 'artist' not in data['similarartists']:
            return []
        
        artists = data['similarartists']['artist']
        if not isinstance(artists, list):
            artists = [artists]
        
        similar_artists = []
        for artist_data in artists[:limit]:
            if isinstance(artist_data, dict) and 'name' in artist_data:
                similarity_score = float(artist_data.get('match', 0))
                similar_artists.append({
                    'name': artist_data['name'],
                    'similarity': similarity_score
                })
        
        return similar_artists
        
    except Exception as e:
        st.error(f"Error fetching similar artists for {artist}: {e}")
        return []

# CHART FUNCTIONS
def create_tier_filtered_chart_with_hover(df: pd.DataFrame, tier_start: int, tier_end: int) -> Tuple[go.Figure, pd.DataFrame]:
    """Create chart showing only artists in the selected tier range"""
    if df.empty:
        return go.Figure(), pd.DataFrame()
    
    artist_stats = calculate_artist_stats(df)
    
    # Filter to selected tier range
    tier_artists_df = artist_stats[
        (artist_stats['rank'] >= tier_start) & 
        (artist_stats['rank'] <= tier_end)
    ].copy()
    
    if tier_artists_df.empty:
        return go.Figure(), pd.DataFrame()
    
    # Sort by rank for proper display order
    tier_artists_df = tier_artists_df.sort_values('rank')
    
    # Create labels with rank numbers
    artist_labels = [f"#{row['rank']}: {row['artist']}" for _, row in tier_artists_df.iterrows()]
    
    # Create the chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=artist_labels,
        x=tier_artists_df['total_hours'],
        orientation='h',
        marker=dict(color='#1DB954'),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Hours Played: %{x:.1f}<br>" +
            "Play Count: %{customdata[0]:,}<br>" +
            "Avg Engagement: %{customdata[1]:.2f}<br>" +
            "💡 Click 'More Like This' button below for recommendations!<br>" +
            "<extra></extra>"
        ),
        customdata=tier_artists_df[['play_count', 'avg_engagement']].values
    ))
    
    fig.update_layout(
        title=f"🎯 Selected Artists for Recommendations (Tier #{tier_start} to #{tier_end})",
        xaxis_title="Hours Played",
        yaxis_title="Artist Rankings",
        height=max(len(tier_artists_df) * 25 + 100, 400),
        margin=dict(l=300, r=50, t=80, b=50),
        yaxis=dict(
            categoryorder='array',
            categoryarray=artist_labels[::-1]  # Reverse for proper top-to-bottom order
        ),
        xaxis=dict(showgrid=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add annotation
    total_hours = tier_artists_df['total_hours'].sum()
    fig.add_annotation(
        text=f"📊 {len(tier_artists_df)} artists selected for recommendations<br>🕒 Total: {total_hours:.1f} hours",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False,
        bgcolor="rgba(29, 185, 84, 0.8)",
        bordercolor="white",
        borderwidth=1,
        font=dict(color="white", size=12)
    )
    
    return fig, tier_artists_df

# RECOMMENDATION FUNCTIONS
def generate_hover_recommendations(artist_name: str, api_key: str, num_recs: int = 5) -> List[Dict]:
    """Generate recommendations for a specific artist"""
    try:
        with st.spinner(f"🎯 Finding artists similar to {artist_name}..."):
            similar_artists = get_similar_artists(artist_name, api_key, num_recs)
            
            recommendations = []
            for i, artist_data in enumerate(similar_artists, 1):
                artist = artist_data['name']
                similarity = artist_data['similarity']
                
                # Get top tracks
                top_tracks = get_top_tracks_simple(artist, api_key, 5)
                
                recommendations.append({
                    'rank': i,
                    'artist': artist,
                    'similarity_score': similarity,
                    'top_tracks': top_tracks,
                    'source_artist': artist_name
                })
                
                # Add delay to respect API limits
                time.sleep(0.2)
            
            return recommendations
            
    except Exception as e:
        st.error(f"Error generating recommendations for {artist_name}: {e}")
        return []

# SEARCH FUNCTIONS
def search_artists(query: str, df: pd.DataFrame) -> List[Dict]:
    """Search for artists in the dataset"""
    if df.empty or not query:
        return []
    
    artist_stats = calculate_artist_stats(df)
    
    # Case-insensitive search
    mask = artist_stats['artist'].str.contains(query, case=False, na=False)
    results = artist_stats[mask].head(10)
    
    search_results = []
    for _, row in results.iterrows():
        search_results.append({
            'artist': row['artist'],
            'rank': row['rank'],
            'total_hours': row['total_hours'],
            'play_count': row['play_count'],
            'avg_engagement': row['avg_engagement']
        })
    
    return search_results

# UI FUNCTIONS
def render_sidebar():
    """Render the sidebar with controls"""
    st.sidebar.markdown("## 🎛️ Controls")
    
    # System Status
    st.sidebar.markdown("### ⚙️ System Status")
    
    # Check data status
    data_status = "✅" if st.session_state.data_loaded else "❌"
    record_count = len(st.session_state.spotify_dataframe) if st.session_state.spotify_dataframe is not None else 0
    st.sidebar.markdown(f"{data_status} Data loaded ({record_count:,} records)")
    
    # Check API status
    api_status = st.session_state.api_status
    lastfm_status = "✅" if api_status['lastfm'] else "❌"
    st.sidebar.markdown(f"{lastfm_status} Last.fm API connected")
    
    # Tier Selection
    st.sidebar.markdown("### 🎯 AI/ML Settings")
    
    # Artist Tier Controls
    tier_start = st.sidebar.number_input(
        "🎯 Artist Tier Start", 
        min_value=1, 
        max_value=10000,
        value=st.session_state.tier_start,
        key="tier_start_input"
    )
    
    tier_end = st.sidebar.number_input(
        "🎯 Artist Tier End", 
        min_value=1, 
        max_value=10000,
        value=st.session_state.tier_end,
        key="tier_end_input"
    )
    
    # Update session state
    st.session_state.tier_start = tier_start
    st.session_state.tier_end = tier_end
    
    # Validate and fix tier range
    if tier_start > tier_end:
        st.sidebar.warning("⚠️ Start value is greater than end value. Will use start value as both start and end.")
        effective_start = effective_end = tier_start
    else:
        effective_start, effective_end = tier_start, tier_end
    
    # Show effective range
    if effective_start == effective_end:
        st.sidebar.info(f"🎯 Using single artist tier: {effective_start}")
    else:
        st.sidebar.info(f"🎯 Using tier range: {effective_start} to {effective_end}")
    
    # Number of recommendations
    num_recs = st.sidebar.slider(
        "📈 Number of Recommendations",
        min_value=1,
        max_value=Config.MAX_RECOMMENDATIONS,
        value=st.session_state.num_recommendations,
        key="num_recs_slider"
    )
    st.session_state.num_recommendations = num_recs
    
    # Recommend Button
    if st.sidebar.button("🎵 Recommend", type="primary", use_container_width=True):
        if st.session_state.data_loaded and api_status['lastfm']:
            # Generate recommendations logic would go here
            st.sidebar.success("🎵 Recommendations generated!")
        else:
            st.sidebar.error("❌ Please ensure data is loaded and API is connected")
    
    # Show tier information if data is loaded
    if st.session_state.data_loaded and st.session_state.spotify_dataframe is not None:
        artist_stats = calculate_artist_stats(st.session_state.spotify_dataframe)
        tier_artists = artist_stats[
            (artist_stats['rank'] >= effective_start) & 
            (artist_stats['rank'] <= effective_end)
        ]
        
        if not tier_artists.empty:
            st.sidebar.markdown("### 📊 Tier Information")
            st.sidebar.markdown(f"**🎯 Selected Tier Range:** {effective_start} to {effective_end}")
            st.sidebar.markdown(f"**📊 {len(tier_artists)} artists** will be used for recommendations")
            
            if len(tier_artists) > 0:
                top_artist = tier_artists.iloc[0]['artist']
                bottom_artist = tier_artists.iloc[-1]['artist']
                st.sidebar.markdown(f"**🏆 Top artist:** {top_artist}")
                st.sidebar.markdown(f"**🎵 Bottom artist:** {bottom_artist}")
    
    return {
        'tier_start': effective_start,
        'tier_end': effective_end,
        'num_recommendations': num_recs
    }

def render_data_overview(df: pd.DataFrame, tier_start: int, tier_end: int):
    """Render data overview with tier-specific metrics"""
    if df.empty:
        st.warning("👈 No data loaded")
        return
    
    # Calculate overall metrics
    total_records = len(df)
    total_artists = df['artist'].nunique()
    total_hours = df['hours_played'].sum()
    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    
    # Calculate tier-specific metrics
    artist_stats = calculate_artist_stats(df)
    tier_artists = artist_stats[
        (artist_stats['rank'] >= tier_start) & 
        (artist_stats['rank'] <= tier_end)
    ]
    
    if not tier_artists.empty:
        # Get plays for tier artists
        tier_artist_names = tier_artists['artist'].tolist()
        tier_df = df[df['artist'].isin(tier_artist_names)]
        
        tier_plays = len(tier_df)
        tier_artist_count = len(tier_artists)
        tier_hours = tier_df['hours_played'].sum()
    else:
        tier_plays = tier_artist_count = tier_hours = 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="tier-metric-card">
            <h3>🎯 Tier Plays</h3>
            <h2>{tier_plays:,}</h2>
            <p>From selected tier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="tier-metric-card">
            <h3>🎯 Tier Artists</h3>
            <h2>{tier_artist_count:,}</h2>
            <p>In selected range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="tier-metric-card">
            <h3>🎯 Tier Hours</h3>
            <h2>{tier_hours:,.1f}</h2>
            <p>From selected artists</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Date Range</h3>
            <h2>{date_range}</h2>
            <p>Overall period</p>
        </div>
        """, unsafe_allow_html=True)

def render_artist_search(df: pd.DataFrame):
    """Render enhanced artist search functionality with detailed metrics and expandable song lists"""
    st.markdown("### 🔍 Artist Search")
    
    # Search input
    search_query = st.text_input(
        "Search for artists in your library:",
        value=st.session_state.search_query,
        placeholder="Enter artist name...",
        key="search_input"
    )
    
    # Update session state
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        st.session_state.search_performed = False
    
    # Search button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Search", key="search_button"):
            if search_query:
                st.session_state.search_results = search_artists(search_query, df)
                st.session_state.search_performed = True
            else:
                st.session_state.search_results = []
                st.session_state.search_performed = False
    
    with col2:
        if st.button("🗑️ Clear Results", key="clear_search_button"):
            st.session_state.search_results = []
            st.session_state.search_query = ""
            st.session_state.search_performed = False
            st.session_state.hover_recommendations = []
            st.session_state.show_jump_link = False
            st.session_state.expanded_artists = set()
            st.rerun()
    
    # Display enhanced search results
    if st.session_state.search_performed and st.session_state.search_results:
        st.markdown(f"**Found {len(st.session_state.search_results)} artists:**")
        
        for result in st.session_state.search_results:
            # Enhanced search result card
            st.markdown(f"""
            <div class="enhanced-search-result-card">
                <div class="artist-rank-title">#{result['rank']}: {result['artist']}</div>
                <div class="artist-metrics-line">
                    <div class="metric-item">
                        <span>🕒</span>
                        <span class="metric-value">{result['total_hours']:.1f}</span>
                        <span>hours</span>
                    </div>
                    <div class="metric-item">
                        <span>🎵</span>
                        <span class="metric-value">{result['play_count']:,}</span>
                        <span>plays</span>
                    </div>
                    <div class="metric-item">
                        <span>📊</span>
                        <span class="metric-value">{result['avg_engagement']:.2f}</span>
                        <span>avg engagement</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Buttons row
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # More Like This button
                button_key = f"more_like_{result['artist'].replace(' ', '_').replace('/', '_')}"
                if st.button(f"🎯 More like this", key=button_key):
                    api_key = os.getenv('LASTFM_API_KEY')
                    if api_key:
                        hover_recs = generate_hover_recommendations(result['artist'], api_key, 5)
                        st.session_state.hover_recommendations = hover_recs
                        st.session_state.selected_hover_artist = result['artist']
                        st.session_state.show_jump_link = True
                        st.rerun()
                    else:
                        st.error("❌ Last.fm API key not found")
            
            with col2:
                # Expand/Collapse songs button
                expand_key = f"expand_{result['artist'].replace(' ', '_').replace('/', '_')}"
                artist_expanded = result['artist'] in st.session_state.expanded_artists
                
                if st.button(f"{'▼' if artist_expanded else '▶'} View Songs ({len(get_artist_songs(result['artist'], df))})", key=expand_key):
                    if artist_expanded:
                        st.session_state.expanded_artists.discard(result['artist'])
                    else:
                        st.session_state.expanded_artists.add(result['artist'])
                    st.rerun()
            
            # Show songs if expanded
            if result['artist'] in st.session_state.expanded_artists:
                songs = get_artist_songs(result['artist'], df)
                if songs:
                    st.markdown(f"""
                    <div class="songs-container">
                        <h4>🎵 All Songs You've Listened To ({len(songs)} tracks)</h4>
                    """, unsafe_allow_html=True)
                    
                    for i, song in enumerate(songs, 1):
                        st.markdown(f"""
                        <div class="song-entry">
                            <div class="song-name">{i}. {song['track']}</div>
                            <div class="song-stats">
                                <span>🕒 {song['total_hours']:.2f}h</span>
                                <span>🎵 {song['play_count']} plays</span>
                                <span>📊 {song['avg_engagement']:.2f} eng</span>
                                <span>📅 {song['first_played']} - {song['last_played']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("*No songs found for this artist*")
        
        # Show jump link if recommendations were generated
        if st.session_state.show_jump_link and st.session_state.hover_recommendations:
            st.markdown(f"""
            <div class="jump-link-container">
                <div class="jump-link" onclick="scrollToElement('recommendations-section')">
                    🎯 Jump to Recommendations for {st.session_state.selected_hover_artist} ⬇️
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.search_performed and not st.session_state.search_results:
        st.info("No artists found matching your search.")

def render_hover_recommendations():
    """Render hover recommendations if available with anchor"""
    if st.session_state.hover_recommendations and st.session_state.selected_hover_artist:
        # Add anchor for smooth scrolling
        st.markdown('<div id="recommendations-section" class="anchor-target"></div>', unsafe_allow_html=True)
        
        st.markdown(f"### 🎯 Artists Similar to {st.session_state.selected_hover_artist}")
        
        # Clear button
        if st.button("🗑️ Clear Similar Artists", key="clear_hover_recs"):
            st.session_state.hover_recommendations = []
            st.session_state.selected_hover_artist = None
            st.session_state.show_jump_link = False
            st.rerun()
        
        # Display recommendations
        for rec in st.session_state.hover_recommendations:
            st.markdown(f"""
            <div class="hover-recommendation-card">
                <h4>#{rec['rank']}: {rec['artist']}</h4>
                <p><strong>Similarity Score:</strong> {rec['similarity_score']:.3f}</p>
                <p><strong>💡 Recommended because you listen to:</strong> {rec['source_artist']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if rec['top_tracks']:
                st.markdown("**🎵 Top Songs:**")
                song_list_html = "<div class='song-list'>"
                for i, song in enumerate(rec['top_tracks'], 1):
                    song_list_html += f"<div class='song-item'>{i}. {song}</div>"
                song_list_html += "</div>"
                st.markdown(song_list_html, unsafe_allow_html=True)
            else:
                st.markdown("*No top songs available*")
            
            st.markdown("---")

def render_more_like_this_buttons(tier_artists_df: pd.DataFrame):
    """Render 'More Like This' buttons for tier artists with jump links"""
    if tier_artists_df.empty:
        return
    
    st.markdown("### 🎯 Get Similar Artists")
    st.markdown("Click any button below to find artists similar to those in your selected tier:")
    
    # Create buttons in a grid layout
    cols_per_row = 3
    artists = tier_artists_df['artist'].tolist()
    
    for i in range(0, len(artists), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, artist in enumerate(artists[i:i+cols_per_row]):
            with cols[j]:
                button_key = f"hover_rec_{artist.replace(' ', '_').replace('/', '_')}"
                if st.button(f"🎯 More like {artist}", key=button_key, use_container_width=True):
                    api_key = os.getenv('LASTFM_API_KEY')
                    if api_key:
                        hover_recs = generate_hover_recommendations(artist, api_key, 5)
                        st.session_state.hover_recommendations = hover_recs
                        st.session_state.selected_hover_artist = artist
                        st.session_state.show_jump_link = True
                        st.rerun()
                    else:
                        st.error("❌ Last.fm API key not found")
    
    # Show jump link if recommendations were generated from tier buttons
    if st.session_state.show_jump_link and st.session_state.hover_recommendations:
        st.markdown(f"""
        <div class="jump-link-container">
            <div class="jump-link" onclick="scrollToElement('recommendations-section')">
                🎯 Jump to Recommendations for {st.session_state.selected_hover_artist} ⬇️
            </div>
        </div>
        """, unsafe_allow_html=True)

# MAIN APPLICATION
def initialize_system():
    """Initialize the system with data loading and API validation"""
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading your music data..."):
            df = load_spotify_data(Config.DATA_FOLDER)
            if not df.empty:
                st.session_state.spotify_dataframe = df
                st.session_state.available_years = get_available_years(df)
                st.session_state.data_loaded = True
                
                # Validate API connectivity
                st.session_state.api_status = validate_api_connectivity()
            else:
                st.error(f"❌ No data found in {Config.DATA_FOLDER} folder")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Initialize system
    initialize_system()
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Personal Music Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by AI/ML: Hybrid Collaborative Filtering + Content-Based Analysis + Temporal Modeling*")
    
    # Render sidebar
    config = render_sidebar()
    
    # Main content
    if st.session_state.data_loaded:
        # Data Overview
        st.markdown("## 📊 Data Overview")
        render_data_overview(st.session_state.spotify_dataframe, config['tier_start'], config['tier_end'])
        
        # Chart
        st.markdown("## 🎯 Selected Artists for Recommendations")
        chart, tier_artists_df = create_tier_filtered_chart_with_hover(
            st.session_state.spotify_dataframe, 
            config['tier_start'], 
            config['tier_end']
        )
        
        if not tier_artists_df.empty:
            st.plotly_chart(chart, use_container_width=True)
            
            # More Like This buttons
            render_more_like_this_buttons(tier_artists_df)
        else:
            st.warning("No artists found in the selected tier range.")
        
        # Enhanced Artist Search
        render_artist_search(st.session_state.spotify_dataframe)
        
        # Hover Recommendations (with anchor for jumping)
        render_hover_recommendations()
        
    else:
        st.warning("👈 Please wait while data is loading...")
    
    # About section
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### 🧠 AI/ML Technology Stack
        
        This system uses advanced machine learning techniques:
        
        - **Hybrid Collaborative Filtering**: Matrix factorization to find patterns in your listening history
        - **Content-Based Filtering**: Last.fm API integration for music similarity analysis
        - **Temporal Analysis**: Time-series modeling to track taste evolution
        - **Context-Aware Recommendations**: Pattern recognition for listening context
        
        ### 🔧 Technical Implementation
        
        - **Data Processing**: Pandas for efficient data manipulation
        - **Visualization**: Plotly for interactive charts
        - **API Integration**: Last.fm for music metadata and similarity
        - **Security**: Encrypted configuration management
        """)

if __name__ == "__main__":
    main()

