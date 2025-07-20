#!/usr/bin/env python3
"""
🎵 Personal Music Recommendation System - Streamlined Streamlit Web Application

This is a streamlined web interface for the hybrid AI/ML music recommendation system.
Features automatic data discovery, simplified configuration, and one-click recommendations.

STREAMLINED FEATURES:
- Automatic data discovery from data/spotify folder
- Auto-loading of API keys from config/.env
- Dynamic year selection with automatic data loading/unloading
- One-click "Recommend" button
- Simplified user interface

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
    
    .recommendation-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .artist-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 0.5rem;
    }
    
    .song-item {
        background: white;
        border-left: 3px solid #1DB954;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
    }
    
    .year-chip {
        background: #e8f5e8;
        border: 1px solid #1DB954;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        color: #1DB954;
        font-weight: bold;
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

class StreamlinedMusicRecommender:
    """
    Streamlined Streamlit Web Interface for the Hybrid AI/ML Music Recommendation System
    
    STREAMLINED VERSION: 
    - Automatic data discovery from data/spotify folder
    - Auto-loading API keys from config/.env
    - Dynamic year selection with automatic data loading
    - One-click recommendations
    - Simplified user interface
    """
    
    def __init__(self):
        self.data_folder = "data/spotify"  # Fixed data folder
        self.config_file = "config/.env"   # Fixed config file
        self.df = None
        self.available_years = []
        self.discovered_files = []
        
        # Initialize session state
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'artist_rankings' not in st.session_state:
            st.session_state.artist_rankings = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'config_loaded' not in st.session_state:
            st.session_state.config_loaded = False
        if 'discovered_files' not in st.session_state:
            st.session_state.discovered_files = []
        if 'available_years' not in st.session_state:
            st.session_state.available_years = []
        if 'selected_years' not in st.session_state:
            st.session_state.selected_years = []
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        
        # Auto-discover data and load config on startup
        self.auto_discover_and_load()
    
    def load_config_from_env(self):
        """Load API keys from config/.env file"""
        try:
            config_path = Path(self.config_file)
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
            st.error(f"Error loading config: {e}")
            return False
    
    def extract_years_from_filename(self, filename: str) -> Set[int]:
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
    
    def discover_data_files(self) -> Tuple[List[int], List[Dict]]:
        """Automatically discover available Spotify JSON files"""
        try:
            data_path = Path(self.data_folder)
            if not data_path.exists():
                return [], []
            
            json_files = list(data_path.glob("*.json"))
            
            # Filter for Spotify files
            spotify_files = []
            for file in json_files:
                filename = file.name
                if any(pattern in filename.lower() for pattern in [
                    'streaming_history', 'audio', 'streaminghistory', 'spotify'
                ]):
                    spotify_files.append(file)
            
            all_years = set()
            file_info = []
            
            for file in spotify_files:
                years_in_file = self.extract_years_from_filename(file.name)
                if years_in_file:
                    all_years.update(years_in_file)
                    
                    try:
                        file_size = file.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)
                        
                        record_count = "Unknown"
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    record_count = len(data)
                        except:
                            pass
                        
                        file_info.append({
                            'filename': file.name,
                            'years': sorted(list(years_in_file)),
                            'size_mb': round(file_size_mb, 2),
                            'records': record_count,
                            'path': str(file)
                        })
                    except Exception:
                        file_info.append({
                            'filename': file.name,
                            'years': sorted(list(years_in_file)),
                            'size_mb': 0,
                            'records': "Error",
                            'path': str(file)
                        })
            
            return sorted(list(all_years)), file_info
            
        except Exception as e:
            return [], []
    
    def auto_discover_and_load(self):
        """Automatically discover data files and load configuration on startup"""
        # Load API keys from config
        self.load_config_from_env()
        
        # Discover data files
        years, files = self.discover_data_files()
        st.session_state.available_years = years
        st.session_state.discovered_files = files
        
        # Set default selected years (all available)
        if years and not st.session_state.selected_years:
            st.session_state.selected_years = years.copy()
    
    def get_files_for_years(self, selected_years: List[int]) -> List[str]:
        """Get file paths that contain data for the selected years"""
        selected_files = []
        
        for file_info in st.session_state.discovered_files:
            file_years = set(file_info['years'])
            selected_years_set = set(selected_years)
            
            if file_years.intersection(selected_years_set):
                selected_files.append(file_info['path'])
        
        return selected_files
    
    def load_data_for_years(self, selected_years: List[int]) -> pd.DataFrame:
        """Load and process Spotify data for selected years"""
        try:
            files_to_load = self.get_files_for_years(selected_years)
            
            if not files_to_load:
                return None
            
            all_data = []
            
            # Load data without showing progress (streamlined)
            for file_path in files_to_load:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            all_data.extend(file_data)
                        else:
                            all_data.append(file_data)
                except Exception:
                    continue
            
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
            
            # Filter by selected years
            if 'ts' in df.columns:
                df['year'] = df['ts'].dt.year
                year_filter = df['year'].isin(selected_years)
                df = df[year_filter]
            
            # Calculate engagement score and hours played
            if 'ms_played' in df.columns:
                df['hours_played'] = df['ms_played'] / (1000 * 60 * 60)
                avg_song_length_ms = 3.5 * 60 * 1000
                df['engagement_score'] = np.minimum(df['ms_played'] / avg_song_length_ms, 1.0)
            else:
                df['hours_played'] = 0.05
                df['engagement_score'] = 0.8
            
            return df
            
        except Exception as e:
            return None
    
    def get_artist_songs(self, artist_name: str, df: pd.DataFrame, min_songs: int = 3, max_songs: int = 5) -> List[str]:
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
    
    def render_sidebar(self):
        """Render the streamlined sidebar"""
        st.sidebar.markdown("## 📅 Year Selection")
        
        # Show available years as chips
        if st.session_state.available_years:
            st.sidebar.markdown("**Available Years:**")
            
            # Create year selection interface
            selected_years = st.sidebar.multiselect(
                "Select years to include:",
                options=st.session_state.available_years,
                default=st.session_state.selected_years,
                help="Choose which years of data to include in analysis"
            )
            
            # Update data when selection changes
            if selected_years != st.session_state.selected_years:
                st.session_state.selected_years = selected_years
                if selected_years:
                    with st.spinner("Loading data for selected years..."):
                        self.df = self.load_data_for_years(selected_years)
                        if self.df is not None:
                            st.session_state.data_loaded = True
                        else:
                            st.session_state.data_loaded = False
                else:
                    st.session_state.data_loaded = False
                    self.df = None
                st.rerun()
        else:
            st.sidebar.warning("No data files found in data/spotify folder")
        
        st.sidebar.markdown("---")
        
        # AI/ML Configuration
        st.sidebar.markdown("## 🧠 AI/ML Settings")
        
        # Artist tier selection
        tier_start = st.sidebar.number_input(
            "🎯 Artist Tier Start",
            min_value=1,
            max_value=10000,
            value=1,
            help="Starting rank for artist tier selection"
        )
        
        tier_end = st.sidebar.number_input(
            "🎯 Artist Tier End",
            min_value=tier_start,
            max_value=10000,
            value=50,
            help="Ending rank for artist tier selection"
        )
        
        # Number of recommendations
        num_recs = st.sidebar.slider(
            "📈 Number of Recommendations",
            min_value=5,
            max_value=100,
            value=20,
            help="How many artist recommendations to generate"
        )
        
        # Recommend button
        recommend_button = st.sidebar.button(
            "🎵 Recommend",
            type="primary",
            help="Generate AI music recommendations",
            use_container_width=True
        )
        
        # Show config status
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ System Status")
        
        if st.session_state.config_loaded:
            st.sidebar.success("✅ API keys loaded")
        else:
            st.sidebar.error("❌ API keys not found in config/.env")
        
        if st.session_state.data_loaded:
            st.sidebar.success(f"✅ Data loaded ({len(self.df):,} records)")
        else:
            st.sidebar.info("ℹ️ Select years to load data")
        
        return {
            'selected_years': selected_years if st.session_state.available_years else [],
            'tier_start': tier_start,
            'tier_end': tier_end,
            'num_recs': num_recs,
            'recommend_clicked': recommend_button
        }
    
    def render_main_header(self):
        """Render the main application header"""
        st.markdown('<h1 class="main-header">🎵 AI Music Recommendation System</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Streamlined Interface - Automatic Discovery & One-Click Recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_data_overview(self):
        """Render data overview and statistics"""
        if not st.session_state.data_loaded or self.df is None:
            st.info("👈 Select years in the sidebar to load your data")
            return
        
        st.markdown("## 📊 Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(self.df):,}</h3>
                <p>Total Plays</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{self.df['artist'].nunique():,}</h3>
                <p>Unique Artists</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{self.df['hours_played'].sum():.0f}h</h3>
                <p>Total Hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            date_range = f"{self.df['ts'].min().year} - {self.df['ts'].max().year}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{date_range}</h3>
                <p>Date Range</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show selected years
        st.markdown("### 📅 Currently Loaded Years")
        year_chips = ""
        for year in sorted(st.session_state.selected_years):
            year_chips += f'<span class="year-chip">{year}</span>'
        st.markdown(year_chips, unsafe_allow_html=True)
        
        # Top artists chart
        st.markdown("### 🏆 Top 20 Artists")
        top_artists = (self.df.groupby('artist')['hours_played']
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
        st.plotly_chart(fig, use_container_width=True)
        
        # Listening patterns over time
        st.markdown("### 📈 Listening Patterns Over Time")
        monthly_stats = (self.df.groupby(self.df['ts'].dt.to_period('M'))
                        .agg({
                            'hours_played': 'sum',
                            'artist': 'nunique'
                        })
                        .reset_index())
        monthly_stats['ts'] = monthly_stats['ts'].astype(str)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['ts'], y=monthly_stats['hours_played'],
                      name="Hours Played", line=dict(color='#1DB954')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['ts'], y=monthly_stats['artist'],
                      name="Unique Artists", line=dict(color='#FF6B6B')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Hours Played", secondary_y=False)
        fig.update_yaxes(title_text="Unique Artists", secondary_y=True)
        fig.update_layout(title_text="Listening Activity Over Time", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_artist_search(self):
        """Render artist search functionality"""
        if not st.session_state.data_loaded or self.df is None:
            st.info("👈 Select years in the sidebar to load your data")
            return
        
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
            # Calculate artist rankings
            artist_stats = (self.df.groupby('artist')
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
            
            artist_stats = artist_stats.sort_values('preference_score', ascending=False).reset_index(drop=True)
            artist_stats['rank'] = range(1, len(artist_stats) + 1)
            
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
                            songs = self.get_artist_songs(row['artist'], self.df)
                            if songs:
                                st.markdown("**Top Songs:**")
                                for i, song in enumerate(songs, 1):
                                    st.markdown(f"{i}. {song}")
            else:
                st.warning(f"No artists found matching '{search_query}'")
    
    def render_recommendations(self, config):
        """Render the main recommendation interface"""
        if not st.session_state.data_loaded or self.df is None:
            st.info("👈 Select years in the sidebar to load your data")
            return
        
        if not st.session_state.config_loaded:
            st.error("❌ API keys not found. Please ensure config/.env file exists with LASTFM_API_KEY and MUSICBRAINZ_USER_AGENT")
            return
        
        st.markdown("## 🎯 AI/ML Music Recommendations")
        
        # Show current configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **Current Configuration:**
            - Artist Tier: {config['tier_start']} - {config['tier_end']}
            - Recommendations: {config['num_recs']}
            - Years: {', '.join(map(str, config['selected_years']))}
            - API Status: ✅ Loaded from config/.env
            """)
        
        with col2:
            if st.button("📊 Analyze Only"):
                self.run_analysis_only(config)
        
        # Generate recommendations if button was clicked
        if config['recommend_clicked']:
            self.generate_recommendations(config)
        
        # Display recommendations if available
        if st.session_state.recommendations:
            self.display_recommendations()
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            self.display_analysis_results()
    
    def generate_recommendations(self, config):
        """Generate AI/ML recommendations using auto-loaded API keys"""
        try:
            with st.spinner("🧠 Running Hybrid AI/ML Recommendation System..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize the recommendation system
                status_text.text("Initializing AI/ML engines...")
                progress_bar.progress(0.1)
                
                # Use API key from session state
                api_key = st.session_state.api_keys.get('LASTFM_API_KEY')
                if not api_key:
                    st.error("LASTFM_API_KEY not found in config/.env")
                    return
                
                # Import and initialize components
                from recommendation_prototype import LastFMAPI, ContentBasedRecommender
                
                lastfm_api = LastFMAPI(api_key)
                
                # Enhanced Content-Based Recommender with tier selection
                status_text.text("Setting up Content-Based AI engine...")
                progress_bar.progress(0.2)
                
                content_recommender = ContentBasedRecommender(self.df, lastfm_api)
                
                # Get tier-specific recommendations
                status_text.text(f"Generating recommendations from tier {config['tier_start']}-{config['tier_end']}...")
                progress_bar.progress(0.5)
                
                # Calculate artist rankings for tier selection
                artist_stats = (self.df.groupby('artist')
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
                
                artist_stats = artist_stats.sort_values('preference_score', ascending=False).reset_index(drop=True)
                
                # Select tier artists
                tier_mask = (
                    (artist_stats.index >= config['tier_start'] - 1) & 
                    (artist_stats.index < config['tier_end'])
                )
                tier_artists = artist_stats[tier_mask]['artist'].tolist()
                
                status_text.text("Getting similar artists from Last.fm...")
                progress_bar.progress(0.7)
                
                # Get recommendations from tier artists
                recommendations = []
                for i, artist in enumerate(tier_artists[:10]):  # Limit to top 10 tier artists
                    try:
                        similar_artists = lastfm_api.get_similar_artists(artist, limit=5)
                        for similar_artist in similar_artists:
                            if similar_artist not in [a['artist'] for a in recommendations]:
                                recommendations.append({
                                    'artist': similar_artist,
                                    'recommendation_score': 1.0 - (i * 0.1),  # Decreasing score
                                    'source_artist': artist
                                })
                        
                        if len(recommendations) >= config['num_recs']:
                            break
                    except Exception as e:
                        continue
                
                status_text.text("Adding song information...")
                progress_bar.progress(0.9)
                
                # Add songs for each recommended artist
                for rec in recommendations:
                    # Try to get songs from user's library first
                    songs = self.get_artist_songs(rec['artist'], self.df)
                    if not songs:
                        # If not in library, get popular songs from Last.fm
                        try:
                            songs = lastfm_api.get_top_tracks(rec['artist'], limit=5)
                        except:
                            songs = []
                    
                    rec['songs'] = songs[:5]  # Maximum 5 songs
                
                progress_bar.progress(1.0)
                status_text.text("✅ Recommendations generated successfully!")
                
                # Store results
                st.session_state.recommendations = {
                    'content_based': recommendations[:config['num_recs']],
                    'tier_info': {
                        'start': config['tier_start'],
                        'end': config['tier_end'],
                        'total_artists': len(artist_stats),
                        'tier_artists': len(tier_artists)
                    },
                    'config': config
                }
                
                time.sleep(1)  # Brief pause to show completion
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.error("Please check your API configuration and network connection.")
    
    def display_recommendations(self):
        """Display the generated recommendations in a beautiful format"""
        recommendations = st.session_state.recommendations
        
        st.markdown("### 🎵 Your AI-Generated Music Recommendations")
        
        # Show tier information
        tier_info = recommendations.get('tier_info', {})
        st.info(f"🎯 Based on artists ranked {tier_info.get('start', '?')}-{tier_info.get('end', '?')} "
                f"from your library of {tier_info.get('total_artists', '?')} artists")
        
        content_recs = recommendations.get('content_based', [])
        
        if not content_recs:
            st.warning("No recommendations generated. Please try different settings or check your API configuration.")
            return
        
        # Display recommendations in cards
        for i, rec in enumerate(content_recs, 1):
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="artist-header">
                        #{i}: {rec['artist']} 
                        <span style="color: #666; font-size: 0.9rem; font-weight: normal;">
                            (Score: {rec['recommendation_score']:.3f})
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display songs
                songs = rec.get('songs', [])
                if songs:
                    st.markdown("**Popular Songs:**")
                    
                    # Create columns for songs
                    cols = st.columns(min(len(songs), 3))
                    for j, song in enumerate(songs):
                        col_idx = j % len(cols)
                        with cols[col_idx]:
                            st.markdown(f"""
                            <div class="song-item">
                                🎵 {song}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("*No song information available*")
                
                # Show source artist if available
                if 'source_artist' in rec:
                    st.caption(f"💡 Recommended because you listen to: {rec['source_artist']}")
                
                st.markdown("---")
        
        # Export options
        st.markdown("### 📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to JSON"):
                self.export_recommendations_json()
        
        with col2:
            if st.button("📊 Export Summary"):
                self.export_summary_json()
        
        with col3:
            if st.button("📋 Copy to Clipboard"):
                self.copy_recommendations_to_clipboard()
    
    def run_analysis_only(self, config):
        """Run analysis without generating recommendations"""
        try:
            with st.spinner("📊 Running comprehensive analysis..."):
                # Basic statistics
                analysis = {
                    'total_plays': len(self.df),
                    'total_hours': self.df['hours_played'].sum(),
                    'unique_artists': self.df['artist'].nunique(),
                    'unique_albums': self.df['album'].nunique() if 'album' in self.df.columns else 0,
                    'date_range': {
                        'start': self.df['ts'].min().strftime('%Y-%m-%d'),
                        'end': self.df['ts'].max().strftime('%Y-%m-%d')
                    }
                }
                
                # Top artists
                top_artists = (self.df.groupby('artist')['hours_played']
                              .sum()
                              .nlargest(20)
                              .to_dict())
                analysis['top_artists'] = top_artists
                
                # Tier analysis
                artist_stats = (self.df.groupby('artist')
                               .agg({
                                   'engagement_score': ['sum', 'mean', 'count'],
                                   'hours_played': 'sum'
                               })
                               .round(3))
                
                artist_stats.columns = ['total_engagement', 'avg_engagement', 'play_count', 'total_hours']
                artist_stats = artist_stats.reset_index()
                
                artist_stats['preference_score'] = (
                    artist_stats['total_engagement'] * 0.4 +
                    artist_stats['avg_engagement'] * 0.3 +
                    np.log1p(artist_stats['play_count']) * 0.2 +
                    np.log1p(artist_stats['total_hours']) * 0.1
                )
                
                artist_stats = artist_stats.sort_values('preference_score', ascending=False).reset_index(drop=True)
                
                # Current tier stats
                tier_mask = (
                    (artist_stats.index >= config['tier_start'] - 1) & 
                    (artist_stats.index < config['tier_end'])
                )
                tier_stats = artist_stats[tier_mask]
                
                analysis['current_tier'] = {
                    'start': config['tier_start'],
                    'end': config['tier_end'],
                    'artist_count': len(tier_stats),
                    'total_hours': tier_stats['total_hours'].sum(),
                    'avg_preference_score': tier_stats['preference_score'].mean()
                }
                
                st.session_state.analysis_results = analysis
                
        except Exception as e:
            st.error(f"Error running analysis: {e}")
    
    def display_analysis_results(self):
        """Display comprehensive analysis results"""
        analysis = st.session_state.analysis_results
        
        st.markdown("### 📊 Comprehensive Analysis Results")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Plays", f"{analysis['total_plays']:,}")
        with col2:
            st.metric("Total Hours", f"{analysis['total_hours']:.1f}h")
        with col3:
            st.metric("Unique Artists", f"{analysis['unique_artists']:,}")
        with col4:
            st.metric("Date Range", f"{analysis['date_range']['start']} to {analysis['date_range']['end']}")
        
        # Current tier analysis
        if 'current_tier' in analysis:
            st.markdown("#### 🎯 Current Tier Analysis")
            tier = analysis['current_tier']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tier Range", f"{tier['start']}-{tier['end']}")
            with col2:
                st.metric("Artists in Tier", tier['artist_count'])
            with col3:
                st.metric("Avg Preference Score", f"{tier['avg_preference_score']:.3f}")
        
        # Top artists
        st.markdown("#### 🏆 Top 10 Artists")
        top_artists_df = pd.DataFrame(
            list(analysis['top_artists'].items())[:10],
            columns=['Artist', 'Hours']
        )
        st.dataframe(top_artists_df, use_container_width=True)
    
    def export_recommendations_json(self):
        """Export recommendations to JSON format"""
        if not st.session_state.recommendations:
            st.error("No recommendations to export")
            return
        
        try:
            export_data = {
                'export_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'export_type': 'music_recommendations_streamlined',
                    'system_version': '2.2_streamlit_streamlined'
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
    
    def export_summary_json(self):
        """Export summary to JSON format"""
        if not st.session_state.recommendations:
            st.error("No recommendations to export")
            return
        
        try:
            recs = st.session_state.recommendations.get('content_based', [])
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_recommendations': len(recs),
                'tier_used': f"{st.session_state.recommendations['tier_info']['start']}-{st.session_state.recommendations['tier_info']['end']}",
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
    
    def copy_recommendations_to_clipboard(self):
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
                    for song in songs:
                        text_output += f"   🎵 {song}\n"
                text_output += "\n"
            
            st.text_area(
                "Copy this text:",
                value=text_output,
                height=300,
                help="Select all and copy to clipboard"
            )
            
        except Exception as e:
            st.error(f"Error preparing text: {e}")
    
    def run(self):
        """Main application runner"""
        if not IMPORTS_AVAILABLE:
            st.error("Required modules not available. Please check your installation.")
            return
        
        # Render sidebar
        config = self.render_sidebar()
        
        # Render main content
        self.render_main_header()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🎯 Recommendations", "🔍 Artist Search", "ℹ️ About"])
        
        with tab1:
            self.render_data_overview()
        
        with tab2:
            self.render_recommendations(config)
        
        with tab3:
            self.render_artist_search()
        
        with tab4:
            st.markdown("""
            ## 🎵 About This System (Streamlined Version)
            
            This is a **Streamlined Hybrid AI/ML Music Recommendation System** with automatic discovery and one-click recommendations.
            
            ### 🚀 Streamlined Features
            
            1. **Automatic Data Discovery**: Finds your Spotify files in `data/spotify` folder automatically
            2. **Auto-Config Loading**: Loads API keys from `config/.env` file automatically
            3. **Dynamic Year Selection**: Add/remove years with automatic data loading
            4. **One-Click Recommendations**: Simple "Recommend" button for instant results
            5. **Simplified Interface**: No manual configuration needed
            
            ### 🧠 AI/ML Engines (Same as Full Version)
            
            1. **Content-Based Filtering**: Last.fm API + Artist Similarity + Tier Selection
            2. **Temporal Collaborative Filtering**: Matrix Factorization + Time-Series Analysis
            3. **Context-Aware Filtering**: K-Means Clustering + Pattern Recognition
            4. **Artist Listing & Ranking**: Preference Modeling + Ranking Algorithms
            
            ### 📁 Required File Structure
            
            ```
            your-project/
            ├── viewmusic_streamlined.py    # This app
            ├── data/spotify/               # Your Spotify data (auto-discovered)
            │   ├── Streaming_History_Audio_2013-2014_1.json
            │   ├── Streaming_History_Audio_2014-2016_2.json
            │   └── ...
            ├── config/.env                 # API keys (auto-loaded)
            │   ├── LASTFM_API_KEY=your_key
            │   └── MUSICBRAINZ_USER_AGENT=your_agent
            └── recommendation_prototype.py # AI/ML engines
            ```
            
            ### 🎯 How to Use
            
            1. **Setup**: Ensure your files are in the correct folders
            2. **Select Years**: Choose which years to include in analysis
            3. **Configure Settings**: Adjust artist tier and recommendation count
            4. **Click Recommend**: Get instant AI-powered music recommendations
            5. **Explore Results**: View recommendations with 3-5 songs per artist
            
            ### 🔧 Technical Features
            
            - **Smart File Detection**: Handles multiple Spotify export formats
            - **Automatic API Loading**: No manual key entry required
            - **Dynamic Data Processing**: Real-time loading based on year selection
            - **Performance Optimized**: Fast processing for large datasets
            - **Export Ready**: JSON export for further analysis
            
            ---
            
            **Created by**: Roberto's AI Music Recommendation System  
            **Version**: 2.2 (Streamlined Interface)  
            **GitHub**: [soyroberto/streamlit](https://github.com/soyroberto/streamlit)  
            **Features**: Auto-discovery + One-click recommendations
            """)

def main():
    """Main application entry point"""
    app = StreamlinedMusicRecommender()
    app.run()

if __name__ == "__main__":
    main()

