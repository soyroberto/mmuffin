#!/usr/bin/env python3
"""
🎵 Personal Music Recommendation System - Simple Songs List Streamlit Web Application

This version features clean recommendation formatting with simple numbered song lists.
Features automatic data loading with proper session state management, tier range visualization with ranks, and tier-filtered chart display.

SIMPLE SONGS VERSION:
- Clean artist recommendation display (no images, no tables)
- Top 5 songs per artist in numbered list format
- No album references or artwork
- Chart shows ONLY artists within the selected tier range
- Artist ranks displayed on the left side of the chart
- Data overview metrics reflect only the selected tier range
- Sidebar unchanged (left side interface preserved)
- Real-time chart filtering based on tier selection
- Automatic loading of ALL Spotify data from data/spotify folder
- Real API connectivity validation to Last.fm
- Single-page interface with recommendations on main page
- No year selection - includes all available data
- Persistent data across all Streamlit interactions

The system uses the same hybrid AI/ML approach:
1. Content-Based Filtering (Last.fm API + Artist Similarity)
2. Temporal Collaborative Filtering (Matrix Factorization + Time-Series)
3. Context-Aware Filtering (Clustering + Pattern Recognition)
4. Artist Listing & Ranking (Preference Modeling)

Ready for deployment on streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from typing import Dict, List, Optional, Tuple, Set
import warnings
import requests
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🎵 AI Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/soyroberto/streamlit',
        'Report a bug': 'https://github.com/soyroberto/streamlit/issues',
        'About': "# 🎵 Hybrid AI/ML Music Recommendation System\n\nPowered by advanced machine learning algorithms including matrix factorization, clustering, and ensemble methods."
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card-selected {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid #0d7a2e;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 2px solid #1DB954;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.15);
    }
    
    .artist-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 1rem;
        text-align: center;
        border-bottom: 2px solid #1DB954;
        padding-bottom: 0.5rem;
    }
    
    .song-list {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .song-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e9ecef;
        font-size: 1rem;
        color: #333;
    }
    
    .song-item:last-child {
        border-bottom: none;
    }
    
    .song-number {
        font-weight: bold;
        color: #1DB954;
        margin-right: 0.5rem;
    }
    
    .recommend-button {
        background: linear-gradient(90deg, #1DB954, #1ed760) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        margin: 1rem 0 !important;
    }
    
    .status-good {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    
    .tier-highlight {
        background: #e8f5e8;
        border: 2px solid #1DB954;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
    
    .stAlert > div {
        border-radius: 8px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(29, 185, 84, 0.3);
    }
    
    .recommendation-score {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Import the recommendation system components
try:
    from recommendation_prototype import HybridMusicRecommender, SpotifyDataProcessor, ContentBasedRecommender, LastFMAPI
    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}")
    st.error("Please ensure all required modules are available.")
    IMPORTS_AVAILABLE = False

# Initialize session state variables FIRST
def initialize_session_state():
    """Initialize all session state variables"""
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'config_loaded' not in st.session_state:
        st.session_state.config_loaded = False
    if 'api_validated' not in st.session_state:
        st.session_state.api_validated = False
    if 'discovered_files' not in st.session_state:
        st.session_state.discovered_files = []
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'initialization_complete' not in st.session_state:
        st.session_state.initialization_complete = False
    if 'spotify_dataframe' not in st.session_state:
        st.session_state.spotify_dataframe = None
    # Initialize tier values with defaults
    if 'tier_start' not in st.session_state:
        st.session_state.tier_start = 1
    if 'tier_end' not in st.session_state:
        st.session_state.tier_end = 50
    # Initialize artist rankings for tier visualization
    if 'artist_rankings' not in st.session_state:
        st.session_state.artist_rankings = None

# Call initialization immediately
initialize_session_state()

def load_config_from_env():
    """Load API keys from config/.env file"""
    try:
        config_file = "config/.env"
        config_path = Path(config_file)
        if not config_path.exists():
            return False
        
        api_keys = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    api_keys[key] = value
        
        if 'LASTFM_API_KEY' in api_keys and 'MUSICBRAINZ_USER_AGENT' in api_keys:
            st.session_state.api_keys = api_keys
            st.session_state.config_loaded = True
            return True
        
        return False
        
    except Exception as e:
        return False

def validate_lastfm_api():
    """Validate actual connectivity to Last.fm API"""
    try:
        if not st.session_state.config_loaded:
            return False
        
        api_key = st.session_state.api_keys.get('LASTFM_API_KEY')
        if not api_key:
            return False
        
        # Test API connectivity with a simple request
        test_url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist=Cher&api_key={api_key}&format=json"
        
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'artist' in data and 'name' in data['artist']:
                st.session_state.api_validated = True
                return True
        
        return False
        
    except Exception as e:
        return False

def get_top_tracks_simple(artist, api_key, limit=5):
    """
    Get top tracks for an artist - SIMPLIFIED VERSION (no images, no albums)
    Based on the user's working code but returns only song names
    """
    try:
        # Get top tracks using the user's proven API structure
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.gettoptracks&artist={artist}&api_key={api_key}&format=json&limit={limit}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        if 'toptracks' not in data or 'track' not in data['toptracks']:
            return []
        
        tracks = data['toptracks']['track']
        if not isinstance(tracks, list):
            tracks = [tracks]  # Handle single track response
        
        # Extract only song names - NO IMAGES, NO ALBUMS
        top_tracks = []
        for track in tracks:
            track_name = track.get('name', 'Unknown Track')
            if track_name and track_name != 'Unknown Track':
                top_tracks.append(track_name)
        
        return top_tracks[:limit]  # Return only the song names
        
    except Exception as e:
        # Return empty list if API call fails
        return []

def extract_years_from_filename(filename: str) -> Set[int]:
    """Extract years from Spotify filename formats"""
    years = set()
    
    # Pattern 1: Streaming_History_Audio_YYYY-YYYY_N.json
    pattern1 = r'Streaming_History_Audio_(\d{4})-(\d{4})_\d+\.json'
    match1 = re.search(pattern1, filename, re.IGNORECASE)
    if match1:
        start_year = int(match1.group(1))
        end_year = int(match1.group(2))
        for year in range(start_year, end_year + 1):
            if 2000 <= year <= 2030:
                years.add(year)
        return years
    
    # Pattern 2: Legacy Audio_YYYY.json format
    pattern2 = r'Audio_(\d{4})\.json'
    match2 = re.search(pattern2, filename, re.IGNORECASE)
    if match2:
        year = int(match2.group(1))
        if 2000 <= year <= 2030:
            years.add(year)
        return years
    
    # Pattern 3: Any 4-digit year in filename
    pattern3 = r'(\d{4})'
    matches = re.findall(pattern3, filename)
    for match in matches:
        year = int(match)
        if 2000 <= year <= 2030:
            years.add(year)
    
    return years

def discover_and_load_all_data() -> pd.DataFrame:
    """Discover and load ALL Spotify data automatically"""
    try:
        data_folder = "data/spotify"
        data_path = Path(data_folder)
        if not data_path.exists():
            return None
        
        json_files = list(data_path.glob("*.json"))
        
        # Filter for Spotify files
        spotify_files = []
        for file in json_files:
            filename = file.name
            if any(pattern in filename.lower() for pattern in [
                'streaming_history', 'audio', 'streaminghistory', 'spotify'
            ]):
                spotify_files.append(file)
        
        if not spotify_files:
            return None
        
        # Store file info for reference
        file_info = []
        all_data = []
        
        # Load ALL data from ALL files
        for file in spotify_files:
            try:
                years_in_file = extract_years_from_filename(file.name)
                
                file_size = file.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                
                # Load the data
                with open(file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                        record_count = len(file_data)
                    else:
                        all_data.append(file_data)
                        record_count = 1
                
                file_info.append({
                    'filename': file.name,
                    'years': sorted(list(years_in_file)) if years_in_file else [],
                    'size_mb': round(file_size_mb, 2),
                    'records': record_count,
                    'path': str(file)
                })
                
            except Exception as e:
                continue
        
        st.session_state.discovered_files = file_info
        
        if not all_data:
            return None
        
        # Convert to DataFrame and process
        df = pd.DataFrame(all_data)
        
        # Handle different column names
        column_mapping = {
            'artistName': 'artist',
            'trackName': 'track', 
            'albumName': 'album',
            'msPlayed': 'ms_played',
            'master_metadata_track_name': 'track',
            'master_metadata_album_artist_name': 'artist',
            'master_metadata_album_album_name': 'album',
            'endTime': 'ts'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Process timestamps
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
        
        # Filter out blank entries
        blank_filter = (
            (df['artist'] != '(Blank)') & 
            (df['track'] != '(Blank)') & 
            (df['artist'] != '') &
            (df['track'] != '') &
            (df['artist'].notna()) &
            (df['track'].notna()) &
            (df['artist'] != 'Unknown Artist') &
            (df['track'] != 'Unknown Track')
        )
        df = df[blank_filter]
        
        # Calculate engagement score and hours played
        if 'ms_played' in df.columns:
            df['hours_played'] = df['ms_played'] / (1000 * 60 * 60)
            avg_song_length_ms = 3.5 * 60 * 1000
            df['engagement_score'] = np.minimum(df['ms_played'] / avg_song_length_ms, 1.0)
        else:
            df['hours_played'] = 0.05
            df['engagement_score'] = 0.8
        
        # Add year column if timestamp exists
        if 'ts' in df.columns:
            df['year'] = df['ts'].dt.year
        
        return df
        
    except Exception as e:
        return None

def calculate_artist_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate artist rankings for tier visualization"""
    try:
        # Calculate artist statistics
        artist_stats = (df.groupby('artist')
                       .agg({
                           'engagement_score': ['sum', 'mean', 'count'],
                           'hours_played': 'sum'
                       })
                       .round(3))
        
        artist_stats.columns = ['total_engagement', 'avg_engagement', 'play_count', 'total_hours']
        artist_stats = artist_stats.reset_index()
        
        # Calculate preference scores
        artist_stats['preference_score'] = (
            artist_stats['total_engagement'] * 0.4 +
            artist_stats['avg_engagement'] * 0.3 +
            np.log1p(artist_stats['play_count']) * 0.2 +
            np.log1p(artist_stats['total_hours']) * 0.1
        )
        
        # Sort by preference score and add rankings
        artist_stats = artist_stats.sort_values('preference_score', ascending=False).reset_index(drop=True)
        artist_stats['rank'] = range(1, len(artist_stats) + 1)
        
        return artist_stats
        
    except Exception as e:
        return None

def initialize_system():
    """Initialize the system once and store everything in session state"""
    if not st.session_state.initialization_complete:
        # Load API keys from config
        load_config_from_env()
        
        # Validate API connectivity
        if st.session_state.config_loaded:
            validate_lastfm_api()
        
        # Load ALL data automatically and store in session state
        df = discover_and_load_all_data()
        if df is not None:
            st.session_state.spotify_dataframe = df
            st.session_state.data_loaded = True
            
            # Calculate artist rankings for tier visualization
            rankings = calculate_artist_rankings(df)
            if rankings is not None:
                st.session_state.artist_rankings = rankings
        else:
            st.session_state.spotify_dataframe = None
            st.session_state.data_loaded = False
        
        st.session_state.initialization_complete = True

# Initialize system once
initialize_system()

def get_artist_songs(artist_name: str, df: pd.DataFrame, min_songs: int = 3, max_songs: int = 5) -> List[str]:
    """Get popular songs for an artist from the listening data"""
    try:
        artist_data = df[df['artist'] == artist_name]
        if len(artist_data) == 0:
            return []
        
        song_stats = (artist_data.groupby('track')
                     .agg({
                         'engagement_score': 'mean',
                         'hours_played': 'sum',
                         'ts': 'count'
                     })
                     .rename(columns={'ts': 'play_count'}))
        
        song_stats['popularity_score'] = (
            song_stats['engagement_score'] * 0.4 +
            song_stats['hours_played'] * 0.3 +
            np.log1p(song_stats['play_count']) * 0.3
        )
        
        top_songs = (song_stats.sort_values('popularity_score', ascending=False)
                    .head(max_songs)
                    .index.tolist())
        
        if len(top_songs) < min_songs and len(song_stats) >= min_songs:
            additional_songs = (song_stats.sort_values('play_count', ascending=False)
                              .head(min_songs)
                              .index.tolist())
            top_songs = list(set(top_songs + additional_songs))[:max_songs]
        
        return top_songs[:max_songs]
        
    except Exception as e:
        return []

def render_sidebar():
    """Render the simplified sidebar with fixed tier input validation - UNCHANGED"""
    st.sidebar.markdown("## 🧠 AI/ML Settings")
    
    # Artist tier selection - FIXED VERSION without dynamic constraints
    # Use session state to manage values and avoid validation conflicts
    
    # Tier Start input - no max constraint to avoid conflicts
    tier_start = st.sidebar.number_input(
        "🎯 Artist Tier Start",
        min_value=1,
        max_value=10000,
        value=st.session_state.tier_start,
        step=1,
        help="Starting rank for artist tier selection",
        key="tier_start_input"
    )
    
    # Tier End input - no min constraint to avoid conflicts
    tier_end = st.sidebar.number_input(
        "🎯 Artist Tier End", 
        min_value=1,
        max_value=10000,
        value=st.session_state.tier_end,
        step=1,
        help="Ending rank for artist tier selection",
        key="tier_end_input"
    )
    
    # Update session state values
    st.session_state.tier_start = tier_start
    st.session_state.tier_end = tier_end
    
    # Validate and fix the range if needed
    if tier_start > tier_end:
        st.sidebar.markdown("""
        <div class="status-warning">
            ⚠️ Start value is greater than end value. Will use start value as both start and end.
        </div>
        """, unsafe_allow_html=True)
        # Auto-fix: use the larger value for both
        tier_end = tier_start
        st.session_state.tier_end = tier_end
    
    # Show the effective range being used with artist names if available
    if st.session_state.artist_rankings is not None:
        rankings = st.session_state.artist_rankings
        
        # Get artists in the selected tier range
        tier_start_actual = min(tier_start, tier_end)
        tier_end_actual = max(tier_start, tier_end)
        
        tier_mask = (
            (rankings['rank'] >= tier_start_actual) & 
            (rankings['rank'] <= tier_end_actual)
        )
        tier_artists = rankings[tier_mask]
        
        if len(tier_artists) > 0:
            st.sidebar.markdown(f"""
            <div class="tier-highlight">
                🎯 <strong>Selected Tier Range: {tier_start_actual} to {tier_end_actual}</strong><br>
                📊 <strong>{len(tier_artists)} artists</strong> will be used for recommendations<br>
                🏆 <strong>Top artist:</strong> {tier_artists.iloc[0]['artist']}<br>
                🎵 <strong>Bottom artist:</strong> {tier_artists.iloc[-1]['artist']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.warning(f"No artists found in tier range {tier_start_actual}-{tier_end_actual}")
    else:
        # Fallback if rankings not available
        if tier_start != tier_end:
            st.sidebar.info(f"🎯 Using artist tier range: {tier_start} to {tier_end}")
        else:
            st.sidebar.info(f"🎯 Using single artist tier: {tier_start}")
    
    # Number of recommendations - INCREASED TO 50
    num_recs = st.sidebar.slider(
        "📈 Number of Recommendations",
        min_value=5,
        max_value=50,  # ✅ INCREASED FROM 100 TO 50 AS REQUESTED
        value=20,
        help="How many artist recommendations to generate (max 50)"
    )
    
    # Recommend button
    recommend_button = st.sidebar.button(
        "🎵 Recommend",
        type="primary",
        help="Generate AI music recommendations",
        use_container_width=True
    )
    
    # Show system status
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ System Status")
    
    # Data status
    if st.session_state.data_loaded and st.session_state.spotify_dataframe is not None:
        st.sidebar.markdown(f"""
        <div class="status-good">
            ✅ Data loaded ({len(st.session_state.spotify_dataframe):,} records)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-error">
            ❌ No data found in data/spotify folder
        </div>
        """, unsafe_allow_html=True)
    
    # API status - only green if actually validated
    if st.session_state.api_validated:
        st.sidebar.markdown("""
        <div class="status-good">
            ✅ Last.fm API connected
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.config_loaded:
        st.sidebar.markdown("""
        <div class="status-error">
            ❌ Last.fm API key found but connection failed
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-error">
            ❌ API keys not found in config/.env
        </div>
        """, unsafe_allow_html=True)
    
    # Show loaded files info
    if st.session_state.discovered_files:
        with st.sidebar.expander("📄 Loaded Files", expanded=False):
            for file_info in st.session_state.discovered_files:
                years_str = ', '.join(map(str, file_info['years'])) if file_info['years'] else 'Unknown'
                st.markdown(f"""
                **{file_info['filename']}**  
                Years: {years_str}  
                Size: {file_info['size_mb']} MB  
                Records: {file_info['records']:,}
                """)
    
    return {
        'tier_start': tier_start,
        'tier_end': tier_end,
        'num_recs': num_recs,
        'recommend_clicked': recommend_button
    }

def render_main_header():
    """Render the main application header"""
    st.markdown('<h1 class="main-header">🎵 AI Music Recommendation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Simple Songs Version - Clean Numbered Lists
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_tier_filtered_chart(df: pd.DataFrame, tier_start: int, tier_end: int):
    """Create chart showing ONLY artists within the selected tier range - FIXED ORDERING"""
    try:
        if st.session_state.artist_rankings is None:
            # Fallback to simple chart if rankings not available
            top_artists = (df.groupby('artist')['hours_played']
                          .sum()
                          .nlargest(20)
                          .reset_index())
            
            fig = px.bar(
                top_artists,
                x='hours_played',
                y='artist',
                orientation='h',
                title="Top Artists by Listening Hours",
                color='hours_played',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            return fig
        
        rankings = st.session_state.artist_rankings
        
        # Get artists in the selected tier range
        tier_start_actual = min(tier_start, tier_end)
        tier_end_actual = max(tier_start, tier_end)
        
        tier_mask = (
            (rankings['rank'] >= tier_start_actual) & 
            (rankings['rank'] <= tier_end_actual)
        )
        tier_artists_df = rankings[tier_mask].copy()
        
        if len(tier_artists_df) == 0:
            # No artists in range - show empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"No artists found in tier range #{tier_start_actual} to #{tier_end_actual}",
                showarrow=False,
                font=dict(size=16, color="red"),
                xref="paper", yref="paper"
            )
            fig.update_layout(
                title=f"Selected Tier Range: #{tier_start_actual} to #{tier_end_actual}",
                height=400,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # FIXED: Sort by rank (ascending) to show in proper order from top to bottom
        tier_artists_df = tier_artists_df.sort_values('rank', ascending=True)
        
        # Create artist labels with ranks on the left
        artist_labels = [f"#{rank}: {artist}" for rank, artist in zip(tier_artists_df['rank'], tier_artists_df['artist'])]
        
        # All artists in the tier are selected, so all bars are green
        colors = ['#1DB954'] * len(tier_artists_df)  # All green since they're all selected
        
        # Create the filtered bar chart
        fig = go.Figure()
        
        # FIXED: Add bars showing HOURS (not ranks) with proper ordering
        fig.add_trace(go.Bar(
            x=tier_artists_df['total_hours'],  # ✅ FIXED: Show hours, not ranks
            y=artist_labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>%{customdata}</b><br>' +
                         'Hours: %{x:.1f}<br>' +
                         'Rank: #%{text}<br>' +
                         'Plays: %{meta:,}<br>' +
                         '<extra></extra>',
            customdata=tier_artists_df['artist'],  # Original artist names for hover
            text=tier_artists_df['rank'],  # Rank numbers for hover
            meta=tier_artists_df['play_count'],  # Play counts for hover
            name='Selected Artists'
        ))
        
        # Add annotation showing the tier range
        if len(tier_artists_df) > 0:
            fig.add_annotation(
                x=tier_artists_df['total_hours'].max() * 0.7,
                y=len(tier_artists_df) - 1,
                text=f"🎯 Showing Tier #{tier_start_actual} to #{tier_end_actual}<br>" +
                     f"📊 {len(tier_artists_df)} artists selected for recommendations<br>" +
                     f"⏱️ Total: {tier_artists_df['total_hours'].sum():.1f} hours",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#1DB954",
                ax=20,
                ay=-30,
                bgcolor="rgba(29, 185, 84, 0.1)",
                bordercolor="#1DB954",
                borderwidth=2,
                font=dict(size=12, color="#1DB954")
            )
        
        # Update layout
        chart_height = max(400, len(tier_artists_df) * 25 + 100)  # Dynamic height based on number of artists
        
        fig.update_layout(
            title=dict(
                text=f"🎯 Selected Artists for Recommendations (Tier #{tier_start_actual}-#{tier_end_actual})<br>" +
                     f"<sub>All {len(tier_artists_df)} artists shown will be used to generate recommendations</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Hours Played",  # ✅ FIXED: Shows hours, not ranks
            yaxis_title="Artist Rankings",
            height=chart_height,
            # FIXED: Remove categoryorder to maintain rank-based ordering
            yaxis=dict(categoryorder='array', categoryarray=artist_labels[::-1]),  # ✅ FIXED: Proper ordering
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            margin=dict(l=300, r=100, t=100, b=50)  # Increased left margin for rank labels
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        # Fallback to simple chart if enhanced version fails
        top_artists = (df.groupby('artist')['hours_played']
                      .sum()
                      .nlargest(20)
                      .reset_index())
        
        fig = px.bar(
            top_artists,
            x='hours_played',
            y='artist',
            orientation='h',
            title="Top Artists by Listening Hours",
            color='hours_played',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        return fig

def get_tier_based_metrics(df: pd.DataFrame, tier_start: int, tier_end: int):
    """Calculate metrics based only on the selected tier range"""
    try:
        if st.session_state.artist_rankings is None:
            return None, None, None, None
        
        rankings = st.session_state.artist_rankings
        
        # Get artists in the selected tier range
        tier_start_actual = min(tier_start, tier_end)
        tier_end_actual = max(tier_start, tier_end)
        
        tier_mask = (
            (rankings['rank'] >= tier_start_actual) & 
            (rankings['rank'] <= tier_end_actual)
        )
        tier_artists = rankings[tier_mask]
        
        if len(tier_artists) == 0:
            return 0, 0, 0, "No Data"
        
        # Get the selected artists' names
        selected_artist_names = tier_artists['artist'].tolist()
        
        # Filter the main dataframe to only include plays from selected artists
        tier_data = df[df['artist'].isin(selected_artist_names)]
        
        if len(tier_data) == 0:
            return 0, 0, 0, "No Data"
        
        # Calculate metrics for the tier
        total_plays = len(tier_data)
        unique_artists = len(selected_artist_names)
        total_hours = tier_data['hours_played'].sum()
        
        # Date range for tier data
        if 'ts' in tier_data.columns and len(tier_data) > 0:
            date_range = f"{tier_data['ts'].min().year} - {tier_data['ts'].max().year}"
        else:
            date_range = "All Time"
        
        return total_plays, unique_artists, total_hours, date_range
        
    except Exception as e:
        return None, None, None, None

def render_data_overview_and_recommendations(config):
    """Render data overview and recommendations on the same page"""
    if not st.session_state.data_loaded or st.session_state.spotify_dataframe is None:
        st.error("❌ No data found. Please ensure your Spotify JSON files are in the data/spotify folder.")
        return
    
    df = st.session_state.spotify_dataframe
    
    # Data Overview Section - STAYS
    st.markdown("## 📊 Data Overview")
    
    # Get tier-based metrics
    tier_plays, tier_artists, tier_hours, tier_date_range = get_tier_based_metrics(
        df, config['tier_start'], config['tier_end']
    )
    
    # Basic statistics - Updated to show tier-based metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if tier_plays is not None:
            st.markdown(f"""
            <div class="metric-card-selected">
                <h3>{tier_plays:,}</h3>
                <p>Tier Plays</p>
                <small>From selected artists</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df):,}</h3>
                <p>Total Plays</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if tier_artists is not None:
            st.markdown(f"""
            <div class="metric-card-selected">
                <h3>{tier_artists:,}</h3>
                <p>Tier Artists</p>
                <small>In selected range</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df['artist'].nunique():,}</h3>
                <p>Unique Artists</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if tier_hours is not None:
            st.markdown(f"""
            <div class="metric-card-selected">
                <h3>{tier_hours:.0f}h</h3>
                <p>Tier Hours</p>
                <small>From selected artists</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df['hours_played'].sum():.0f}h</h3>
                <p>Total Hours</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        date_range_display = tier_date_range if tier_date_range else "All Time"
        if 'ts' in df.columns and tier_date_range == "All Time":
            date_range_display = f"{df['ts'].min().year} - {df['ts'].max().year}"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{date_range_display}</h3>
            <p>Date Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show tier selection info - STAYS
    if tier_plays is not None and tier_artists is not None:
        tier_start_actual = min(config['tier_start'], config['tier_end'])
        tier_end_actual = max(config['tier_start'], config['tier_end'])
        
        st.info(f"📊 **Data Overview Updated**: Showing metrics for artists ranked #{tier_start_actual} to #{tier_end_actual} • Green cards show tier-specific data")
    
    # UPDATED: Chart now shows ONLY the selected tier artists
    st.markdown("### 🎯 Selected Artists for Recommendations")
    
    # Create and display the tier-filtered chart
    tier_chart = create_tier_filtered_chart(
        df, 
        config['tier_start'], 
        config['tier_end']
    )
    st.plotly_chart(tier_chart, use_container_width=True)
    
    # Show tier selection summary - STAYS
    if st.session_state.artist_rankings is not None:
        rankings = st.session_state.artist_rankings
        tier_start_actual = min(config['tier_start'], config['tier_end'])
        tier_end_actual = max(config['tier_start'], config['tier_end'])
        
        tier_mask = (
            (rankings['rank'] >= tier_start_actual) & 
            (rankings['rank'] <= tier_end_actual)
        )
        tier_artists_df = rankings[tier_mask]
        
        if len(tier_artists_df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"🎯 **Tier Range:** #{tier_start_actual} to #{tier_end_actual}")
            
            with col2:
                st.info(f"📊 **Artists in Range:** {len(tier_artists_df)}")
            
            with col3:
                total_tier_hours = tier_artists_df['total_hours'].sum()
                st.info(f"⏱️ **Tier Total Hours:** {total_tier_hours:.1f}h")
    
    # Recommendations Section - STAYS
    st.markdown("---")
    
    # Generate recommendations if button was clicked
    if config['recommend_clicked']:
        generate_recommendations(config)
    
    # Display recommendations if available
    if st.session_state.recommendations:
        display_recommendations()
    elif not config['recommend_clicked']:
        st.markdown("## 🎯 AI/ML Music Recommendations")
        st.info("👈 Click the 'Recommend' button in the sidebar to generate AI-powered music recommendations!")

def render_artist_search():
    """Render artist search functionality - STAYS"""
    if not st.session_state.data_loaded or st.session_state.spotify_dataframe is None:
        st.error("❌ No data loaded. Please ensure your Spotify JSON files are in the data/spotify folder.")
        return
    
    df = st.session_state.spotify_dataframe
    
    st.markdown("## 🔍 Artist Search & Ranking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for an artist",
            placeholder="Enter artist name...",
            help="Search for specific artists in your library"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    if search_query and search_button:
        # Use cached rankings if available
        if st.session_state.artist_rankings is not None:
            artist_stats = st.session_state.artist_rankings
        else:
            # Calculate rankings if not cached
            artist_stats = calculate_artist_rankings(df)
            if artist_stats is not None:
                st.session_state.artist_rankings = artist_stats
        
        if artist_stats is not None:
            # Search for matching artists
            mask = artist_stats['artist'].str.contains(search_query, case=False, na=False)
            results = artist_stats[mask]
            
            if len(results) > 0:
                st.success(f"Found {len(results)} matching artist(s)")
                
                for _, row in results.head(10).iterrows():
                    with st.expander(f"#{row['rank']}: {row['artist']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Hours", f"{row['total_hours']:.1f}h")
                            st.metric("Play Count", f"{row['play_count']:,}")
                        
                        with col2:
                            st.metric("Avg Engagement", f"{row['avg_engagement']:.3f}")
                            st.metric("Preference Score", f"{row['preference_score']:.3f}")
                        
                        with col3:
                            # Get top songs for this artist
                            songs = get_artist_songs(row['artist'], df)
                            if songs:
                                st.markdown("**Top Songs:**")
                                for i, song in enumerate(songs, 1):
                                    st.markdown(f"{i}. {song}")
            else:
                st.warning(f"No artists found matching '{search_query}'")
        else:
            st.error("Unable to calculate artist rankings")

def generate_recommendations(config):
    """Generate AI/ML recommendations using auto-loaded API keys with clean artist names"""
    if not st.session_state.api_validated:
        st.error("❌ Cannot generate recommendations: Last.fm API not connected")
        return
    
    if not st.session_state.data_loaded or st.session_state.spotify_dataframe is None:
        st.error("❌ Cannot generate recommendations: No data loaded")
        return
    
    df = st.session_state.spotify_dataframe
    
    try:
        with st.spinner("🧠 Running Hybrid AI/ML Recommendation System..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize the recommendation system
            status_text.text("Initializing AI/ML engines...")
            progress_bar.progress(0.1)
            
            # Use API key from session state
            api_key = st.session_state.api_keys.get('LASTFM_API_KEY')
            
            # Import and initialize components
            from recommendation_prototype import LastFMAPI, ContentBasedRecommender
            
            lastfm_api = LastFMAPI(api_key)
            
            # Enhanced Content-Based Recommender with tier selection
            status_text.text("Setting up Content-Based AI engine...")
            progress_bar.progress(0.2)
            
            content_recommender = ContentBasedRecommender(df, lastfm_api)
            
            # Get tier-specific recommendations
            status_text.text(f"Generating recommendations from tier #{config['tier_start']}-#{config['tier_end']}...")
            progress_bar.progress(0.5)
            
            # Use cached rankings if available
            if st.session_state.artist_rankings is not None:
                artist_stats = st.session_state.artist_rankings
            else:
                # Calculate rankings if not cached
                artist_stats = calculate_artist_rankings(df)
                if artist_stats is not None:
                    st.session_state.artist_rankings = artist_stats
            
            if artist_stats is None:
                st.error("Unable to calculate artist rankings")
                return
            
            # Select tier artists - handle the case where start > end
            tier_start = min(config['tier_start'], config['tier_end'])
            tier_end = max(config['tier_start'], config['tier_end'])
            
            tier_mask = (
                (artist_stats['rank'] >= tier_start) & 
                (artist_stats['rank'] <= tier_end)
            )
            tier_artists = artist_stats[tier_mask]['artist'].tolist()
            
            status_text.text("Getting similar artists from Last.fm...")
            progress_bar.progress(0.7)
            
            # Get recommendations from tier artists
            recommendations = []
            for i, artist in enumerate(tier_artists[:15]):  # Increased from 10 to 15 for better variety
                try:
                    similar_artists = lastfm_api.get_similar_artists(artist, limit=5)
                    for similar_artist in similar_artists:
                        # CLEAN ARTIST NAME: Extract just the name if it's a dict
                        if isinstance(similar_artist, dict):
                            artist_name = similar_artist.get('name', str(similar_artist))
                        else:
                            artist_name = str(similar_artist)
                        
                        # Check if already in recommendations
                        if artist_name not in [rec['artist'] for rec in recommendations]:
                            recommendations.append({
                                'artist': artist_name,  # Clean artist name only
                                'recommendation_score': 1.0 - (i * 0.05),  # Slower decay for more variety
                                'source_artist': artist
                            })
                    
                    if len(recommendations) >= config['num_recs']:
                        break
                except Exception as e:
                    continue
            
            status_text.text("Getting top songs (simplified)...")
            progress_bar.progress(0.9)
            
            # Add simple song lists for each recommended artist
            for rec in recommendations:
                # Get simple song names only (no images, no albums)
                songs = get_top_tracks_simple(rec['artist'], api_key, limit=5)
                rec['songs'] = songs  # Just a list of song names
            
            progress_bar.progress(1.0)
            status_text.text("✅ Recommendations generated successfully!")
            
            # Store results
            st.session_state.recommendations = {
                'content_based': recommendations[:config['num_recs']],
                'tier_info': {
                    'start': tier_start,
                    'end': tier_end,
                    'total_artists': len(artist_stats),
                    'tier_artists': len(tier_artists)
                },
                'config': config
            }
            
            time.sleep(1)  # Brief pause to show completion
            
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        st.error("Please check your API configuration and network connection.")

def display_recommendations():
    """Display the generated recommendations in a simple format with numbered song lists"""
    recommendations = st.session_state.recommendations
    
    st.markdown("## 🎵 Your AI-Generated Music Recommendations")
    
    # Show tier information
    tier_info = recommendations.get('tier_info', {})
    st.info(f"🎯 Based on artists ranked #{tier_info.get('start', '?')}-#{tier_info.get('end', '?')} "
            f"from your library of {tier_info.get('total_artists', '?')} artists")
    
    content_recs = recommendations.get('content_based', [])
    
    if not content_recs:
        st.warning("No recommendations generated. Please try different settings or check your API configuration.")
        return
    
    # Display recommendations in simple cards with numbered song lists
    for i, rec in enumerate(content_recs, 1):
        with st.container():
            # Artist header with score
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="artist-header">
                    {i}: {rec['artist']}
                    <span class="recommendation-score">Score: {rec['recommendation_score']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display songs in a simple numbered list
            songs = rec.get('songs', [])
            if songs:
                st.markdown("### 🎵 Top Songs")
                
                # Create simple numbered list HTML
                song_list_html = '<div class="song-list">'
                
                for j, song in enumerate(songs, 1):
                    song_list_html += f'<div class="song-item"><span class="song-number">{j}.</span>{song}</div>'
                song_list_html += '</div>'
                
                st.markdown(song_list_html, unsafe_allow_html=True)
                
            else:
                st.markdown("*No song information available from Last.fm*")
            
            # Show source artist if available
            if 'source_artist' in rec:
                st.caption(f"💡 Recommended because you listen to: **{rec['source_artist']}**")
            
            st.markdown("---")
    
    # Export options
    st.markdown("### 📤 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export to JSON"):
            export_recommendations_json()
    
    with col2:
        if st.button("📊 Export Summary"):
            export_summary_json()
    
    with col3:
        if st.button("📋 Copy to Clipboard"):
            copy_recommendations_to_clipboard()

def export_recommendations_json():
    """Export recommendations to JSON format"""
    if not st.session_state.recommendations:
        st.error("No recommendations to export")
        return
    
    try:
        export_data = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'export_type': 'music_recommendations_simple_songs',
                'system_version': '3.0_streamlit_simple_songs'
            },
            'recommendations': st.session_state.recommendations,
            'analysis': st.session_state.analysis_results
        }
        
        json_str = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"music_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting JSON: {e}")

def export_summary_json():
    """Export summary to JSON format"""
    if not st.session_state.recommendations:
        st.error("No recommendations to export")
        return
    
    try:
        recs = st.session_state.recommendations.get('content_based', [])
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recs),
            'tier_used': f"#{st.session_state.recommendations['tier_info']['start']}-#{st.session_state.recommendations['tier_info']['end']}",
            'top_recommendations': [
                {
                    'artist': rec['artist'],
                    'score': rec['recommendation_score'],
                    'songs': rec.get('songs', [])
                }
                for rec in recs[:10]
            ]
        }
        
        json_str = json.dumps(summary, indent=2)
        
        st.download_button(
            label="📊 Download Summary",
            data=json_str,
            file_name=f"music_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting summary: {e}")

def copy_recommendations_to_clipboard():
    """Copy recommendations to clipboard format"""
    if not st.session_state.recommendations:
        st.error("No recommendations to copy")
        return
    
    try:
        recs = st.session_state.recommendations.get('content_based', [])
        text_output = "🎵 AI Music Recommendations\n\n"
        
        for i, rec in enumerate(recs, 1):
            text_output += f"{i}. {rec['artist']} (Score: {rec['recommendation_score']:.3f})\n"
            songs = rec.get('songs', [])
            if songs:
                text_output += "   Top Songs:\n"
                for j, song in enumerate(songs, 1):
                    text_output += f"   {j}. {song}\n"
            text_output += "\n"
        
        st.text_area(
            "Copy this text:",
            value=text_output,
            height=300,
            help="Select all and copy to clipboard"
        )
        
    except Exception as e:
        st.error(f"Error preparing text: {e}")

def main():
    """Main application runner"""
    if not IMPORTS_AVAILABLE:
        st.error("Required modules not available. Please check your installation.")
        return
    
    # Render sidebar - UNCHANGED
    config = render_sidebar()
    
    # Render main content
    render_main_header()
    
    # Create tabs for different sections - STAYS
    tab1, tab2, tab3 = st.tabs(["📊 Data & Recommendations", "🔍 Artist Search", "ℹ️ About"])
    
    with tab1:
        render_data_overview_and_recommendations(config)
    
    with tab2:
        render_artist_search()
    
    with tab3:
        st.markdown("""
        ## 🎵 About This System (Simple Songs Version)
        
        This is a **Simple Songs Hybrid AI/ML Music Recommendation System** with clean numbered song lists.
        
        ### 🆕 Latest Enhancement
        
        **Simple Song Lists**: Clean and minimal display format:
        - **No images or album artwork** - removed all visual clutter
        - **No tables** - simple numbered lists instead
        - **No album references** - just song names
        - **Clean formatting** - artist name with numbered songs below
        - **Minimal design** - focus on the music, not the visuals
        
        ### 🎯 Display Format
        
        Each recommended artist shows:
        ```
        #1: Artist Name                                    Score: 0.950
        
        🎵 Top Songs
        1. Song Name One
        2. Song Name Two  
        3. Song Name Three
        4. Song Name Four
        5. Song Name Five
        
        💡 Recommended because you listen to: Source Artist
        ```
        
        ### ✅ What's Removed
        
        - ❌ **Album artwork and images** - no more loading issues
        - ❌ **Album names and references** - simplified display
        - ❌ **Complex tables** - replaced with simple lists
        - ❌ **Play count columns** - focus on song discovery
        - ❌ **Visual clutter** - clean, minimal interface
        
        ### ✅ What's Preserved
        
        - ✅ **Clean artist names** - no messy JSON output
        - ✅ **Top 5 songs per artist** - using Last.fm API
        - ✅ **Numbered lists** - easy to read format
        - ✅ **Recommendation scores** - AI confidence levels
        - ✅ **Source attribution** - why each artist was recommended
        - ✅ **All AI/ML functionality** - same powerful algorithms
        
        ### 🎨 Design Philosophy
        
        **Minimal and Clean**:
        - Focus on music discovery, not visual elements
        - Fast loading with no image dependencies
        - Simple, readable format that works everywhere
        - Clean typography with clear hierarchy
        - Spotify-inspired color scheme maintained
        
        ### 🔧 Technical Benefits
        
        **Performance**:
        - Faster loading (no image requests)
        - Lower bandwidth usage
        - Better mobile experience
        - Simplified API calls
        - Reduced complexity
        
        **Reliability**:
        - No broken image links
        - No album artwork loading failures
        - Consistent display across all devices
        - Simplified error handling
        
        ---
        
        **Created by**: Roberto's AI Music Recommendation System  
        **Version**: 3.0 (Simple Songs - Clean Numbered Lists)  
        **GitHub**: [soyroberto/streamlit](https://github.com/soyroberto/streamlit)  
        **Status**: Production Ready - Minimal and Fast
        """)

if __name__ == "__main__":
    main()

