#!/usr/bin/env python3
"""
Music Recommendation System Prototype
Based on 11 years of Spotify listening history

This prototype demonstrates the core components of a hybrid music recommendation system
that combines temporal collaborative filtering, content-based similarity, and context-aware recommendations.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

class SpotifyDataProcessor:
    """Process raw Spotify listening history data"""
    
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load all Spotify JSON files and combine into single DataFrame"""
        dataframes = []
        
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.json') and 'Audio' in filename:
                filepath = os.path.join(self.data_directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        df_temp = pd.DataFrame(data)
                        dataframes.append(df_temp)
                        print(f"Loaded {len(df_temp)} records from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
            self._clean_data()
            return self.df
        else:
            raise ValueError("No valid data files found")
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        # Convert timestamp
        self.df['ts'] = pd.to_datetime(self.df['ts'])
        
        # Calculate engagement metrics
        self.df['hours_played'] = self.df['ms_played'] / 3600000
        self.df['minutes_played'] = self.df['ms_played'] / 60000
        
        # Extract temporal features
        self.df['year'] = self.df['ts'].dt.year
        self.df['month'] = self.df['ts'].dt.month
        self.df['day_of_week'] = self.df['ts'].dt.dayofweek
        self.df['hour'] = self.df['ts'].dt.hour
        
        # Clean artist and track names
        self.df['artist'] = self.df['master_metadata_album_artist_name'].fillna('Unknown Artist')
        self.df['track'] = self.df['master_metadata_track_name'].fillna('Unknown Track')
        self.df['album'] = self.df['master_metadata_album_album_name'].fillna('Unknown Album')
        
        # Calculate engagement score (0-1 based on listening completion)
        # Assume average track length of 3.5 minutes for missing data
        avg_track_length_ms = 3.5 * 60 * 1000
        self.df['engagement_score'] = np.minimum(
            self.df['ms_played'] / avg_track_length_ms, 1.0
        )
        
        # Remove very short plays (likely skips)
        self.df = self.df[self.df['ms_played'] > 30000]  # At least 30 seconds
        
        print(f"Processed {len(self.df)} listening records")
        print(f"Date range: {self.df['ts'].min()} to {self.df['ts'].max()}")
        print(f"Unique artists: {self.df['artist'].nunique()}")
        print(f"Unique albums: {self.df['album'].nunique()}")

class LastFMAPI:
    """Interface to Last.fm API for music metadata and similarity"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        self.session = requests.Session()
        
    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[Dict]:
        """Get similar artists from Last.fm"""
        params = {
            'method': 'artist.getsimilar',
            'artist': artist_name,
            'api_key': self.api_key,
            'format': 'json',
            'limit': limit
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'similarartists' in data and 'artist' in data['similarartists']:
                return data['similarartists']['artist']
            return []
        except Exception as e:
            print(f"Error fetching similar artists for {artist_name}: {e}")
            return []
    
    def get_artist_info(self, artist_name: str) -> Dict:
        """Get artist information from Last.fm"""
        params = {
            'method': 'artist.getinfo',
            'artist': artist_name,
            'api_key': self.api_key,
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'artist' in data:
                return data['artist']
            return {}
        except Exception as e:
            print(f"Error fetching artist info for {artist_name}: {e}")
            return {}

class TemporalCollaborativeFilter:
    """Time-aware collaborative filtering for taste evolution"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.time_periods = self._create_time_periods()
        self.artist_embeddings = {}
        
    def _create_time_periods(self) -> List[Tuple[str, pd.DataFrame]]:
        """Split data into time periods for temporal analysis"""
        periods = []
        
        # Yearly periods
        for year in sorted(self.df['year'].unique()):
            year_data = self.df[self.df['year'] == year]
            if len(year_data) > 100:  # Minimum threshold
                periods.append((f"Year_{year}", year_data))
        
        # Recent periods (more granular)
        recent_data = self.df[self.df['year'] >= 2020]
        if len(recent_data) > 500:
            for year in [2020, 2021, 2022, 2023]:
                year_data = recent_data[recent_data['year'] == year]
                if len(year_data) > 100:
                    periods.append((f"Recent_{year}", year_data))
        
        return periods
    
    def train_period_embeddings(self):
        """Train embeddings for each time period"""
        for period_name, period_data in self.time_periods:
            # Create artist-engagement matrix for this period
            artist_engagement = (period_data.groupby('artist')['engagement_score']
                               .agg(['sum', 'mean', 'count'])
                               .reset_index())
            
            # Calculate weighted engagement score
            artist_engagement['weighted_score'] = (
                artist_engagement['sum'] * 0.5 +
                artist_engagement['mean'] * 0.3 +
                np.log1p(artist_engagement['count']) * 0.2
            )
            
            self.artist_embeddings[period_name] = artist_engagement
            print(f"Processed {len(artist_engagement)} artists for {period_name}")
    
    def get_taste_evolution(self) -> Dict:
        """Analyze how musical taste evolves over time"""
        evolution = {}
        
        for period_name, embeddings in self.artist_embeddings.items():
            top_artists = embeddings.nlargest(10, 'weighted_score')['artist'].tolist()
            evolution[period_name] = top_artists
        
        return evolution
    
    def predict_future_preferences(self, recent_weight: float = 0.7) -> List[str]:
        """Predict future preferences based on taste evolution"""
        if not self.artist_embeddings:
            self.train_period_embeddings()
        
        # Weight recent periods more heavily
        weighted_scores = {}
        
        for period_name, embeddings in self.artist_embeddings.items():
            weight = recent_weight if 'Recent' in period_name else (1 - recent_weight)
            
            for _, row in embeddings.iterrows():
                artist = row['artist']
                score = row['weighted_score'] * weight
                
                if artist in weighted_scores:
                    weighted_scores[artist] += score
                else:
                    weighted_scores[artist] = score
        
        # Return top predicted artists
        sorted_artists = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return [artist for artist, score in sorted_artists[:50]]

class ContentBasedRecommender:
    """Content-based recommendations using artist similarity"""
    
    def __init__(self, df: pd.DataFrame, lastfm_api: Optional[LastFMAPI] = None):
        self.df = df
        self.lastfm_api = lastfm_api
        self.user_profile = self._build_user_profile()
        
    def _build_user_profile(self) -> Dict:
        """Build user profile from listening history"""
        # Calculate artist preferences
        artist_stats = (self.df.groupby('artist')
                       .agg({
                           'engagement_score': ['sum', 'mean', 'count'],
                           'hours_played': 'sum'
                       })
                       .round(3))
        
        artist_stats.columns = ['total_engagement', 'avg_engagement', 'play_count', 'total_hours']
        artist_stats = artist_stats.reset_index()
        
        # Calculate preference score
        artist_stats['preference_score'] = (
            artist_stats['total_engagement'] * 0.4 +
            artist_stats['avg_engagement'] * 0.3 +
            np.log1p(artist_stats['play_count']) * 0.2 +
            np.log1p(artist_stats['total_hours']) * 0.1
        )
        
        # Get top artists
        top_artists = artist_stats.nlargest(20, 'preference_score')
        
        return {
            'top_artists': top_artists['artist'].tolist(),
            'artist_scores': dict(zip(artist_stats['artist'], artist_stats['preference_score']))
        }
    
    def get_similar_artists_recommendations(self, num_recommendations: int = 20) -> List[Dict]:
        """Get artist recommendations based on similarity to user's favorites"""
        if not self.lastfm_api:
            print("Last.fm API not available for similarity recommendations")
            return []
        
        recommendations = {}
        
        # Get similar artists for each of user's top artists
        for artist in self.user_profile['top_artists'][:10]:  # Top 10 to avoid API limits
            similar_artists = self.lastfm_api.get_similar_artists(artist, limit=10)
            
            for similar_artist in similar_artists:
                if isinstance(similar_artist, dict) and 'name' in similar_artist:
                    similar_name = similar_artist['name']
                    similarity_score = float(similar_artist.get('match', 0))
                    
                    # Skip if already in user's listening history
                    if similar_name in self.user_profile['artist_scores']:
                        continue
                    
                    # Weight by user's preference for the seed artist
                    seed_preference = self.user_profile['artist_scores'].get(artist, 0)
                    weighted_score = similarity_score * seed_preference
                    
                    if similar_name in recommendations:
                        recommendations[similar_name] += weighted_score
                    else:
                        recommendations[similar_name] = weighted_score
            
            # Rate limiting
            time.sleep(0.2)
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for artist_name, score in sorted_recs[:num_recommendations]:
            result.append({
                'artist': artist_name,
                'recommendation_score': score,
                'type': 'similar_artist'
            })
        
        return result

class ContextAwareRecommender:
    """Context-aware recommendations based on listening patterns"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.context_profiles = self._build_context_profiles()
    
    def _build_context_profiles(self) -> Dict:
        """Build context-specific listening profiles"""
        profiles = {}
        
        # Time-of-day profiles
        for hour_range, label in [
            (range(6, 12), 'morning'),
            (range(12, 18), 'afternoon'),
            (range(18, 23), 'evening'),
            (list(range(23, 24)) + list(range(0, 6)), 'night')
        ]:
            hour_data = self.df[self.df['hour'].isin(hour_range)]
            if len(hour_data) > 50:
                top_artists = (hour_data.groupby('artist')['engagement_score']
                             .agg(['sum', 'count'])
                             .reset_index())
                top_artists['score'] = top_artists['sum'] * np.log1p(top_artists['count'])
                profiles[f'time_{label}'] = top_artists.nlargest(10, 'score')['artist'].tolist()
        
        # Day-of-week profiles
        for dow, label in [(0, 'monday'), (5, 'saturday'), (6, 'sunday')]:
            dow_data = self.df[self.df['day_of_week'] == dow]
            if len(dow_data) > 50:
                top_artists = (dow_data.groupby('artist')['engagement_score']
                             .agg(['sum', 'count'])
                             .reset_index())
                top_artists['score'] = top_artists['sum'] * np.log1p(top_artists['count'])
                profiles[f'day_{label}'] = top_artists.nlargest(10, 'score')['artist'].tolist()
        
        return profiles
    
    def get_context_recommendations(self, context: str) -> List[str]:
        """Get recommendations for specific context"""
        if context in self.context_profiles:
            return self.context_profiles[context]
        else:
            return []

class HybridMusicRecommender:
    """Main recommendation system combining multiple approaches"""
    
    def __init__(self, data_directory: str, lastfm_api_key: Optional[str] = None):
        # Initialize components
        self.data_processor = SpotifyDataProcessor(data_directory)
        self.df = self.data_processor.load_data()
        
        self.lastfm_api = LastFMAPI(lastfm_api_key) if lastfm_api_key else None
        
        self.temporal_cf = TemporalCollaborativeFilter(self.df)
        self.content_recommender = ContentBasedRecommender(self.df, self.lastfm_api)
        self.context_recommender = ContextAwareRecommender(self.df)
        
        print("Hybrid Music Recommender initialized successfully!")
    
    def get_comprehensive_recommendations(self, num_recommendations: int = 20) -> Dict:
        """Get recommendations from all engines"""
        recommendations = {}
        
        # Temporal collaborative filtering
        print("Generating temporal recommendations...")
        temporal_recs = self.temporal_cf.predict_future_preferences()
        recommendations['temporal'] = temporal_recs[:num_recommendations]
        
        # Content-based recommendations
        print("Generating content-based recommendations...")
        content_recs = self.content_recommender.get_similar_artists_recommendations(num_recommendations)
        recommendations['content_based'] = content_recs
        
        # Context-aware recommendations
        print("Generating context-aware recommendations...")
        context_recs = {}
        for context in ['time_morning', 'time_evening', 'day_saturday']:
            context_recs[context] = self.context_recommender.get_context_recommendations(context)
        recommendations['context_aware'] = context_recs
        
        return recommendations
    
    def analyze_listening_patterns(self) -> Dict:
        """Analyze user's listening patterns and preferences"""
        analysis = {}
        
        # Basic statistics
        analysis['total_plays'] = len(self.df)
        analysis['total_hours'] = self.df['hours_played'].sum()
        analysis['unique_artists'] = self.df['artist'].nunique()
        analysis['unique_albums'] = self.df['album'].nunique()
        analysis['date_range'] = {
            'start': self.df['ts'].min().strftime('%Y-%m-%d'),
            'end': self.df['ts'].max().strftime('%Y-%m-%d')
        }
        
        # Top artists and albums
        top_artists = (self.df.groupby('artist')['hours_played']
                      .sum()
                      .nlargest(10)
                      .to_dict())
        analysis['top_artists'] = top_artists
        
        top_albums = (self.df.groupby(['artist', 'album'])['hours_played']
                     .sum()
                     .nlargest(10)
                     .to_dict())
        analysis['top_albums'] = {f"{artist} - {album}": hours 
                                 for (artist, album), hours in top_albums.items()}
        
        # Temporal patterns
        hourly_listening = self.df.groupby('hour')['hours_played'].sum().to_dict()
        analysis['hourly_patterns'] = hourly_listening
        
        # Taste evolution
        taste_evolution = self.temporal_cf.get_taste_evolution()
        analysis['taste_evolution'] = taste_evolution
        
        return analysis

# Example usage and testing
if __name__ == "__main__":
    # Example usage (requires actual data directory and API key)
    print("Initializing Music Recommendation System Prototype...")
    # Initialize the recommender system
    recommender = HybridMusicRecommender(
        data_directory="/Users/roberto/OneDrive/Azure/Spotify/MyData2",
        lastfm_api_key="6e885bb2f90e58a1d3968aed8b173e5c"  # Replace with your Last.fm API key
    )
    
    # Analyze listening patterns
    analysis = recommender.analyze_listening_patterns()
    print("Listening Analysis:")
    print(f"Total plays: {analysis['total_plays']}")
    print(f"Total hours: {analysis['total_hours']:.1f}")
    print(f"Unique artists: {analysis['unique_artists']}")
    
    # Get recommendations
    recommendations = recommender.get_comprehensive_recommendations(num_recommendations=15)
    
    print("\nTemporal Recommendations:")
    for artist in recommendations['temporal'][:10]:
        print(f"- {artist}")
    
    print("\nContent-Based Recommendations:")
    for rec in recommendations['content_based'][:10]:
        print(f"- {rec['artist']} (score: {rec['recommendation_score']:.3f})")
    
    print("\nContext-Aware Recommendations (Evening):")
    for artist in recommendations['context_aware'].get('time_evening', [])[:5]:
        print(f"- {artist}")

    print("Music Recommendation System Prototype")
    print("=====================================")
    print("This prototype demonstrates a hybrid approach combining:")
    print("1. Temporal Collaborative Filtering")
    print("2. Content-Based Similarity (via Last.fm API)")
    print("3. Context-Aware Recommendations")
    print("4. Comprehensive Analysis of Listening Patterns")
    print("\nTo use this system:")
    print("1. Place your Spotify data JSON files in a directory")
    print("2. Get a Last.fm API key (optional but recommended)")
    print("3. Initialize the HybridMusicRecommender class")
    print("4. Call get_comprehensive_recommendations() for results")

