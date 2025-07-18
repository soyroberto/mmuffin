#!/usr/bin/env python3
"""
🎵 Personal Music Recommendation System - Streamlit Web Application

This is a comprehensive web interface for the hybrid AI/ML music recommendation system.
It provides all the functionality of the CLI version (ymusic.py) in an interactive web format.

Features:
- Interactive artist tier selection
- Year-based data filtering
- AI/ML recommendations with song details
- Artist search and listing
- JSON export functionality
- Real-time analysis and visualization
- Responsive design for all devices

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
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from typing import Dict, List, Optional, Tuple
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
    from secrets_encryption_system import SecureConfigManager, SecureConfigLoader
    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}")
    st.error("Please ensure all required modules are available.")
    IMPORTS_AVAILABLE = False

class StreamlitMusicRecommender:
    """
    Streamlit Web Interface for the Hybrid AI/ML Music Recommendation System
    
    This class provides a web-based interface for all the functionality available
    in the CLI version, with enhanced visualization and interactivity.
    """
    
    def __init__(self):
        self.data_processor = None
        self.recommender_system = None
        self.df = None
        self.available_years = []
        self.config_loaded = False
        
        # Initialize session state
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'artist_rankings' not in st.session_state:
            st.session_state.artist_rankings = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'config_setup' not in st.session_state:
            st.session_state.config_setup = False
    
    def load_configuration(self):
        """Load secure configuration for API access"""
        try:
            config_manager = SecureConfigManager()
            config_loader = SecureConfigLoader(config_manager)
            
            # Try different configuration methods
            for method in ['env', 'encrypted', 'prompt']:
                try:
                    secrets = config_loader.load_secrets(method)
                    if secrets and secrets.get('LASTFM_API_KEY'):
                        st.session_state.secrets = secrets
                        st.session_state.config_setup = True
                        return True
                except:
                    continue
            
            return False
        except Exception as e:
            st.error(f"Configuration loading error: {e}")
            return False
    
    def discover_data_files(self, data_folder: str) -> List[str]:
        """Discover available Spotify JSON files and extract years"""
        try:
            data_path = Path(data_folder)
            if not data_path.exists():
                return []
            
            json_files = list(data_path.glob("*.json"))
            audio_files = [f for f in json_files if 'Audio' in f.name or 'StreamingHistory' in f.name]
            
            years = []
            for file in audio_files:
                # Extract year from filename (e.g., Audio_2012.json -> 2012)
                filename = file.stem
                for part in filename.split('_'):
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                        if 2000 <= year <= 2030:  # Reasonable year range
                            years.append(year)
                            break
            
            return sorted(list(set(years)))
        except Exception as e:
            st.error(f"Error discovering data files: {e}")
            return []
    
    def load_data_for_years(self, data_folder: str, selected_years: List[int]) -> pd.DataFrame:
        """Load and process Spotify data for selected years"""
        try:
            data_path = Path(data_folder)
            all_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, year in enumerate(selected_years):
                status_text.text(f"Loading data for {year}...")
                
                # Look for files containing the year
                year_files = []
                for file in data_path.glob("*.json"):
                    if str(year) in file.name:
                        year_files.append(file)
                
                for file in year_files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            year_data = json.load(f)
                            if isinstance(year_data, list):
                                all_data.extend(year_data)
                            else:
                                all_data.append(year_data)
                    except Exception as e:
                        st.warning(f"Could not load {file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(selected_years))
            
            status_text.text("Processing data...")
            
            if not all_data:
                st.error("No data found for selected years")
                return None
            
            # Convert to DataFrame and process
            df = pd.DataFrame(all_data)
            
            # Data cleaning and feature engineering
            if 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'])
            elif 'endTime' in df.columns:
                df['ts'] = pd.to_datetime(df['endTime'])
                df = df.rename(columns={'endTime': 'ts'})
            
            # Handle different column names from different Spotify export formats
            if 'artistName' in df.columns:
                df = df.rename(columns={'artistName': 'artist'})
            if 'trackName' in df.columns:
                df = df.rename(columns={'trackName': 'track'})
            if 'albumName' in df.columns:
                df = df.rename(columns={'albumName': 'album'})
            if 'msPlayed' in df.columns:
                df = df.rename(columns={'msPlayed': 'ms_played'})
            
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
            
            # Calculate engagement score and hours played
            if 'ms_played' in df.columns:
                df['hours_played'] = df['ms_played'] / (1000 * 60 * 60)  # Convert to hours
                # Estimate engagement score (assuming average song length of 3.5 minutes)
                avg_song_length_ms = 3.5 * 60 * 1000
                df['engagement_score'] = np.minimum(df['ms_played'] / avg_song_length_ms, 1.0)
            else:
                # Fallback if ms_played not available
                df['hours_played'] = 0.05  # Assume 3 minutes per play
                df['engagement_score'] = 0.8  # Assume good engagement
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Loaded {len(df):,} listening records")
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def get_artist_songs(self, artist_name: str, df: pd.DataFrame, min_songs: int = 3, max_songs: int = 5) -> List[str]:
        """Get popular songs for an artist from the listening data"""
        try:
            artist_data = df[df['artist'] == artist_name]
            if len(artist_data) == 0:
                return []
            
            # Get top songs by play count and engagement
            song_stats = (artist_data.groupby('track')
                         .agg({
                             'engagement_score': 'mean',
                             'hours_played': 'sum',
                             'ts': 'count'  # play count
                         })
                         .rename(columns={'ts': 'play_count'}))
            
            # Calculate song popularity score
            song_stats['popularity_score'] = (
                song_stats['engagement_score'] * 0.4 +
                song_stats['hours_played'] * 0.3 +
                np.log1p(song_stats['play_count']) * 0.3
            )
            
            # Get top songs
            top_songs = (song_stats.sort_values('popularity_score', ascending=False)
                        .head(max_songs)
                        .index.tolist())
            
            # Ensure we have at least min_songs if available
            if len(top_songs) < min_songs and len(song_stats) >= min_songs:
                additional_songs = (song_stats.sort_values('play_count', ascending=False)
                                  .head(min_songs)
                                  .index.tolist())
                top_songs = list(set(top_songs + additional_songs))[:max_songs]
            
            return top_songs[:max_songs]
            
        except Exception as e:
            st.error(f"Error getting songs for {artist_name}: {e}")
            return []
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.markdown("## 🎛️ Configuration")
        
        # Data folder selection
        data_folder = st.sidebar.text_input(
            "📁 Data Folder Path",
            value="data/spotify",
            help="Path to your Spotify JSON files"
        )
        
        # Discover available years
        if st.sidebar.button("🔍 Discover Data Files"):
            with st.spinner("Discovering data files..."):
                self.available_years = self.discover_data_files(data_folder)
                if self.available_years:
                    st.sidebar.success(f"Found data for years: {', '.join(map(str, self.available_years))}")
                else:
                    st.sidebar.error("No data files found. Please check the folder path.")
        
        # Year selection
        if self.available_years:
            selected_years = st.sidebar.multiselect(
                "📅 Select Years",
                options=self.available_years,
                default=self.available_years,
                help="Choose which years of data to include in analysis"
            )
        else:
            selected_years = []
            st.sidebar.info("Click 'Discover Data Files' to see available years")
        
        # Load data button
        if st.sidebar.button("📊 Load Data") and selected_years:
            with st.spinner("Loading and processing data..."):
                self.df = self.load_data_for_years(data_folder, selected_years)
                if self.df is not None:
                    st.session_state.data_loaded = True
                    st.session_state.selected_years = selected_years
                    st.sidebar.success(f"✅ Loaded {len(self.df):,} records")
        
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
        
        # API Configuration
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🔐 API Configuration")
        
        api_key = st.sidebar.text_input(
            "🔑 Last.fm API Key",
            type="password",
            help="Your Last.fm API key for external music data"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            st.session_state.config_setup = True
        
        return {
            'data_folder': data_folder,
            'selected_years': selected_years,
            'tier_start': tier_start,
            'tier_end': tier_end,
            'num_recs': num_recs,
            'api_key': api_key
        }
    
    def render_main_header(self):
        """Render the main application header"""
        st.markdown('<h1 class="main-header">🎵 AI Music Recommendation System</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Powered by Hybrid AI/ML algorithms including Matrix Factorization, 
                Clustering, and Ensemble Learning
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_data_overview(self):
        """Render data overview and statistics"""
        if not st.session_state.data_loaded or self.df is None:
            st.info("👆 Please load your data using the sidebar to get started")
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
            return
        
        if not st.session_state.config_setup:
            st.warning("⚠️ Please configure your Last.fm API key in the sidebar to get recommendations")
            return
        
        st.markdown("## 🎯 AI/ML Music Recommendations")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            **Configuration:**
            - Artist Tier: {config['tier_start']} - {config['tier_end']}
            - Recommendations: {config['num_recs']}
            - Years: {', '.join(map(str, config['selected_years']))}
            """)
        
        with col2:
            generate_button = st.button("🧠 Generate AI Recommendations", type="primary")
        
        with col3:
            analyze_button = st.button("📊 Analyze Only")
        
        if generate_button:
            self.generate_recommendations(config)
        
        if analyze_button:
            self.run_analysis_only(config)
        
        # Display recommendations if available
        if st.session_state.recommendations:
            self.display_recommendations()
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            self.display_analysis_results()
    
    def generate_recommendations(self, config):
        """Generate AI/ML recommendations"""
        try:
            with st.spinner("🧠 Running Hybrid AI/ML Recommendation System..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize the recommendation system
                status_text.text("Initializing AI/ML engines...")
                progress_bar.progress(0.1)
                
                # Import and initialize components
                from recommendation_prototype import LastFMAPI, ContentBasedRecommender
                
                lastfm_api = LastFMAPI(config['api_key'])
                
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
            st.error("Please check your API key and network connection.")
    
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
            st.warning("No recommendations generated. Please try different settings or check your API key.")
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
                    'export_type': 'music_recommendations_web',
                    'system_version': '2.0_streamlit'
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
            ## 🎵 About This System
            
            This is a **Hybrid AI/ML Music Recommendation System** that combines multiple machine learning approaches:
            
            ### 🧠 AI/ML Engines
            
            1. **Content-Based Filtering**
               - Uses Last.fm API for external music knowledge
               - Implements cosine similarity for artist matching
               - Features artist tier selection for targeted recommendations
            
            2. **Temporal Collaborative Filtering**
               - Applies Non-Negative Matrix Factorization (NMF)
               - Analyzes listening patterns over time
               - Predicts future preferences using time-series analysis
            
            3. **Context-Aware Filtering**
               - Uses K-Means clustering for context discovery
               - Considers time of day, day of week, and seasonal patterns
               - Provides context-specific recommendations
            
            4. **Artist Listing & Ranking**
               - Implements preference modeling with weighted scoring
               - Provides fast artist search and filtering
               - Supports tier-based exploration
            
            ### 📊 Mathematical Foundation
            
            - **Preference Score**: `P = 0.4×engagement + 0.3×avg_engagement + 0.2×log(plays) + 0.1×log(hours)`
            - **Matrix Factorization**: `V ≈ W × H` (NMF decomposition)
            - **Ensemble Scoring**: Weighted combination of multiple models
            
            ### 🔧 Technology Stack
            
            - **Frontend**: Streamlit (Python web framework)
            - **ML Libraries**: scikit-learn, numpy, pandas
            - **Visualization**: Plotly
            - **APIs**: Last.fm for external music data
            - **Security**: Encrypted configuration management
            
            ### 🚀 Features
            
            - Interactive year-based data filtering
            - Real-time AI/ML recommendations
            - Artist search with song details
            - Comprehensive data analysis
            - JSON export functionality
            - Responsive web design
            - Ready for streamlit.io deployment
            
            ---
            
            **Created by**: Roberto's AI Music Recommendation System  
            **Version**: 2.0 (Streamlit Web Interface)  
            **GitHub**: [soyroberto/streamlit](https://github.com/soyroberto/streamlit)
            """)

def main():
    """Main application entry point"""
    app = StreamlitMusicRecommender()
    app.run()

if __name__ == "__main__":
    main()

