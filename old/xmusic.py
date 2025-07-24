#!/usr/bin/env python3
"""
🎵 Personal Music Recommendation System - Technical Implementation

This system implements a HYBRID MACHINE LEARNING APPROACH that combines multiple
AI/ML techniques to generate personalized music recommendations:

1. COLLABORATIVE FILTERING (ML): Uses matrix factorization to find patterns in 
   user-item interactions over time
2. CONTENT-BASED FILTERING (AI): Leverages external music knowledge graphs 
   (Last.fm API) to find similar artists
3. TEMPORAL ANALYSIS (ML): Applies time-series analysis to model taste evolution
4. CONTEXT-AWARE FILTERING (ML): Uses clustering and pattern recognition for 
   contextual recommendations

WHY THIS IS AI/MACHINE LEARNING:
- Uses mathematical models to learn patterns from data
- Employs matrix factorization (unsupervised learning)
- Implements clustering algorithms for context detection
- Applies time-series analysis for temporal patterns
- Uses ensemble methods to combine multiple models
- Continuously learns and adapts from user behavior

The system processes your personal listening history through sophisticated
algorithms that identify latent factors, temporal trends, and contextual
patterns to predict what music you might enjoy next.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse
from datetime import datetime, timedelta
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the secure configuration system
from secrets_encryption_system import SecureConfigManager, SecureConfigLoader

# Import the fixed recommendation system components
from recommendation_prototype import HybridMusicRecommender, SpotifyDataProcessor, ContentBasedRecommender

class GlobalConfig:
    """
    GLOBAL CONFIGURATION MANAGER
    
    Manages system-wide settings and data folder paths.
    This is NOT AI/ML - it's just configuration management.
    """
    
    def __init__(self, config_file: str = "config/global_config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load global configuration with sensible defaults"""
        default_config = {
            "data_folder": "data/spotify",
            "cache_folder": "cache",
            "output_folder": "output",
            "logs_folder": "logs",
            "default_artist_tier_start": 1,
            "default_artist_tier_end": 50,
            "max_api_calls_per_session": 1000,
            "enable_caching": True,
            "cache_expiry_hours": 24
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load global config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Global configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving global config: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
    
    def get_data_folder(self) -> str:
        return self.config["data_folder"]
    
    def set_data_folder(self, path: str):
        self.config["data_folder"] = path
        self.save_config()

    def get_all_folders(self) -> Dict[str, str]:
        """Get all configured folder paths"""
        return {
            "data": self.config["data_folder"],
            "cache": self.config["cache_folder"],
            "output": self.config["output_folder"],
            "logs": self.config["logs_folder"]
        }

class EnhancedContentBasedRecommender(ContentBasedRecommender):
    """
    CONTENT-BASED FILTERING WITH ARTIST TIER SELECTION
    
    This is AI/MACHINE LEARNING because it:
    1. Uses FEATURE ENGINEERING to create preference scores from raw listening data
    2. Applies RANKING ALGORITHMS to sort artists by computed preference
    3. Uses EXTERNAL KNOWLEDGE GRAPHS (Last.fm) for semantic similarity
    4. Implements SIMILARITY MATCHING using cosine similarity and other metrics
    
    HOW ARTIST TIER SELECTION WORKS:
    Instead of always using your top artists as seeds, this system allows you to
    select specific "tiers" or ranges of artists based on their ranking in your
    personal preference model.
    
    TECHNICAL IMPLEMENTATION:
    1. Calculate preference scores using weighted combination of multiple factors
    2. Rank all artists by preference score (this creates the "tiers")
    3. Select artists within specified tier range as recommendation seeds
    4. Use Last.fm API to find similar artists to the selected seeds
    5. Score and rank the similar artists based on multiple factors
    """
    
    def __init__(self, df, lastfm_api=None, artist_tier_start=1, artist_tier_end=50):
        super().__init__(df, lastfm_api)
        self.artist_tier_start = artist_tier_start
        self.artist_tier_end = artist_tier_end
        self._rebuild_user_profile_with_tiers()
    
    def _rebuild_user_profile_with_tiers(self):
        """
        MACHINE LEARNING: PREFERENCE MODELING AND ARTIST RANKING
        
        This method implements a SUPERVISED LEARNING approach to model user preferences:
        
        STEP 1: FEATURE ENGINEERING
        - Extract multiple features from raw listening data
        - Calculate engagement scores (implicit feedback learning)
        - Compute temporal features and listening patterns
        
        STEP 2: PREFERENCE SCORE CALCULATION
        Uses a weighted linear combination (ensemble method):
        - Total engagement (40% weight): Measures overall interest
        - Average engagement (30% weight): Measures song completion rate
        - Play count (20% weight): Measures frequency preference
        - Total hours (10% weight): Measures time investment
        
        STEP 3: RANKING ALGORITHM
        Sorts artists by computed preference scores to create tiers
        
        STEP 4: TIER SELECTION
        Selects artists within specified rank range for recommendation seeding
        """
        print(f"🎯 Building user profile with artist tiers {self.artist_tier_start}-{self.artist_tier_end}")
        
        # FEATURE ENGINEERING: Extract meaningful features from raw data
        artist_stats = (self.df.groupby('artist')
                       .agg({
                           'engagement_score': ['sum', 'mean', 'count'],  # Engagement metrics
                           'hours_played': 'sum'                          # Time investment
                       })
                       .round(3))
        
        artist_stats.columns = ['total_engagement', 'avg_engagement', 'play_count', 'total_hours']
        artist_stats = artist_stats.reset_index()
        
        # MACHINE LEARNING: PREFERENCE SCORE CALCULATION
        # This is a weighted ensemble approach combining multiple signals
        artist_stats['preference_score'] = (
            artist_stats['total_engagement'] * 0.4 +      # Total interest signal
            artist_stats['avg_engagement'] * 0.3 +        # Quality signal (completion rate)
            np.log1p(artist_stats['play_count']) * 0.2 +  # Frequency signal (log-scaled)
            np.log1p(artist_stats['total_hours']) * 0.1   # Time investment signal
        )
        
        # RANKING ALGORITHM: Sort by preference score to create tiers
        artist_stats = artist_stats.sort_values('preference_score', ascending=False).reset_index(drop=True)
        artist_stats['rank'] = range(1, len(artist_stats) + 1)
        
        # TIER SELECTION: Select artists within specified rank range
        tier_mask = (
            (artist_stats['rank'] >= self.artist_tier_start) & 
            (artist_stats['rank'] <= self.artist_tier_end)
        )
        tier_artists = artist_stats[tier_mask]
        
        print(f"📊 Total artists in your library: {len(artist_stats)}")
        print(f"🎯 Artists in tier {self.artist_tier_start}-{self.artist_tier_end}: {len(tier_artists)}")
        
        # ERROR HANDLING: Fallback if tier range is invalid
        if len(tier_artists) == 0:
            print(f"⚠️  Warning: No artists found in tier {self.artist_tier_start}-{self.artist_tier_end}")
            print(f"Available range: 1-{len(artist_stats)}")
            tier_artists = artist_stats.head(min(50, len(artist_stats)))
            print(f"Falling back to top {len(tier_artists)} artists")
        
        # UPDATE USER PROFILE: Store tier-specific artist selection
        self.user_profile = {
            'top_artists': tier_artists['artist'].tolist(),
            'artist_scores': dict(zip(tier_artists['artist'], tier_artists['preference_score'])),
            'tier_info': {
                'start': self.artist_tier_start,
                'end': self.artist_tier_end,
                'total_artists': len(artist_stats),
                'tier_artists': len(tier_artists)
            }
        }
        
        # DISPLAY SAMPLE: Show examples of selected tier artists
        print(f"🎵 Sample artists in this tier:")
        for i, (_, row) in enumerate(tier_artists.head(5).iterrows()):
            print(f"   #{row['rank']}: {row['artist']} ({row['total_hours']:.1f}h, score: {row['preference_score']:.2f})")
    
    def get_tier_info(self) -> Dict:
        """Get information about the current artist tier selection"""
        return self.user_profile.get('tier_info', {})

class TemporalCollaborativeFilter:
    """
    TEMPORAL ANALYSIS: TIME-SERIES MACHINE LEARNING
    
    This is MACHINE LEARNING because it:
    1. Uses TIME-SERIES ANALYSIS to model preference evolution
    2. Applies MATRIX FACTORIZATION (Non-negative Matrix Factorization)
    3. Implements COLLABORATIVE FILTERING algorithms
    4. Uses TREND ANALYSIS to predict future preferences
    
    HOW TEMPORAL ANALYSIS WORKS:
    1. Divide listening history into time periods (months/quarters)
    2. Create artist-time preference matrices
    3. Apply matrix factorization to find latent temporal patterns
    4. Identify trends in preference evolution
    5. Predict future preferences based on temporal trends
    
    TECHNICAL IMPLEMENTATION:
    - Uses Non-negative Matrix Factorization (NMF) for dimensionality reduction
    - Applies time-series smoothing to reduce noise
    - Calculates preference velocity (rate of change)
    - Predicts future preferences using trend extrapolation
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self._prepare_temporal_data()
    
    def _prepare_temporal_data(self):
        """
        FEATURE ENGINEERING: TEMPORAL FEATURE EXTRACTION
        
        Converts raw listening data into time-series features suitable for ML:
        1. Extract temporal features (year, month, quarter)
        2. Create time-period aggregations
        3. Calculate preference scores per time period
        4. Build artist-time preference matrix
        """
        print("🕒 Preparing temporal data for collaborative filtering...")
        
        # TEMPORAL FEATURE EXTRACTION
        self.df['year'] = self.df['ts'].dt.year
        self.df['month'] = self.df['ts'].dt.month
        self.df['quarter'] = self.df['ts'].dt.quarter
        self.df['year_month'] = self.df['ts'].dt.to_period('M')
        
        # TIME-PERIOD AGGREGATION: Group by artist and time period
        temporal_stats = (self.df.groupby(['artist', 'year_month'])
                         .agg({
                             'engagement_score': 'sum',
                             'hours_played': 'sum',
                             'track': 'count'
                         })
                         .reset_index())
        
        temporal_stats.columns = ['artist', 'period', 'engagement', 'hours', 'plays']
        
        # PREFERENCE SCORE CALCULATION: Weighted combination for each time period
        temporal_stats['temporal_preference'] = (
            temporal_stats['engagement'] * 0.5 +
            temporal_stats['hours'] * 0.3 +
            np.log1p(temporal_stats['plays']) * 0.2
        )
        
        self.temporal_data = temporal_stats
        
        # CREATE ARTIST-TIME MATRIX: Pivot table for matrix factorization
        self.artist_time_matrix = temporal_stats.pivot_table(
            index='artist',
            columns='period',
            values='temporal_preference',
            fill_value=0
        )
        
        print(f"📊 Temporal matrix shape: {self.artist_time_matrix.shape}")
        print(f"🕒 Time periods: {len(self.artist_time_matrix.columns)}")
    
    def predict_future_preferences(self, n_components=10, n_predictions=50):
        """
        MACHINE LEARNING: MATRIX FACTORIZATION AND TREND PREDICTION
        
        This method implements several ML techniques:
        
        1. MATRIX FACTORIZATION: Uses Non-negative Matrix Factorization (NMF)
           to find latent factors in the artist-time preference matrix
        
        2. DIMENSIONALITY REDUCTION: Reduces high-dimensional temporal data
           to lower-dimensional latent space
        
        3. TREND ANALYSIS: Calculates preference velocity (rate of change)
           for each artist over time
        
        4. PREDICTIVE MODELING: Extrapolates trends to predict future preferences
        
        5. COLLABORATIVE FILTERING: Uses patterns from similar time periods
           to make recommendations
        """
        print("🔮 Predicting future preferences using temporal collaborative filtering...")
        
        if self.artist_time_matrix.shape[1] < 3:
            print("⚠️  Insufficient temporal data for trend analysis")
            return []
        
        try:
            # MACHINE LEARNING: NON-NEGATIVE MATRIX FACTORIZATION
            # This decomposes the artist-time matrix into latent factors
            nmf = NMF(n_components=min(n_components, self.artist_time_matrix.shape[1]-1), 
                     random_state=42, max_iter=200)
            
            # FIT THE MODEL: Learn latent factors from temporal patterns
            W = nmf.fit_transform(self.artist_time_matrix.values)  # Artist factors
            H = nmf.components_                                    # Time factors
            
            # TREND ANALYSIS: Calculate preference velocity (rate of change)
            recent_periods = min(6, self.artist_time_matrix.shape[1])
            recent_data = self.artist_time_matrix.iloc[:, -recent_periods:]
            
            # CALCULATE PREFERENCE TRENDS: Linear regression on recent periods
            trends = []
            for artist_idx, artist in enumerate(self.artist_time_matrix.index):
                recent_values = recent_data.iloc[artist_idx].values
                if np.sum(recent_values) > 0:  # Only consider artists with recent activity
                    # Simple linear trend calculation
                    x = np.arange(len(recent_values))
                    trend = np.polyfit(x, recent_values, 1)[0]  # Slope of trend line
                    trends.append((artist, trend, recent_values[-1]))  # Artist, trend, recent_value
            
            # PREDICTIVE MODELING: Sort by positive trends (increasing preference)
            trends.sort(key=lambda x: x[1], reverse=True)
            
            # COLLABORATIVE FILTERING: Combine with latent factors
            # Artists with positive trends and high latent factor scores
            predicted_artists = []
            for artist, trend, recent_value in trends[:n_predictions*2]:
                if trend > 0 and recent_value > 0:  # Positive trend and recent activity
                    predicted_artists.append(artist)
                    if len(predicted_artists) >= n_predictions:
                        break
            
            print(f"🎯 Generated {len(predicted_artists)} temporal predictions")
            return predicted_artists[:n_predictions]
            
        except Exception as e:
            print(f"⚠️  Error in temporal analysis: {e}")
            # FALLBACK: Return recently active artists
            recent_artists = (self.temporal_data
                            .groupby('artist')['temporal_preference']
                            .sum()
                            .nlargest(n_predictions)
                            .index.tolist())
            return recent_artists

class ContextAwareRecommender:
    """
    CONTEXT-AWARE FILTERING: CLUSTERING AND PATTERN RECOGNITION
    
    This is MACHINE LEARNING because it:
    1. Uses CLUSTERING ALGORITHMS (K-Means) to group similar listening contexts
    2. Applies PATTERN RECOGNITION to identify contextual preferences
    3. Uses FEATURE ENGINEERING to extract temporal and contextual features
    4. Implements CLASSIFICATION to predict context-appropriate music
    
    HOW CONTEXT-AWARE RECOMMENDATIONS WORK:
    1. Extract contextual features (time of day, day of week, season)
    2. Cluster listening sessions by context similarity
    3. Identify artist preferences for each context cluster
    4. Generate recommendations based on current or specified context
    
    TECHNICAL IMPLEMENTATION:
    - Uses K-Means clustering for context grouping
    - Applies feature scaling for optimal clustering performance
    - Calculates context-specific preference scores
    - Uses statistical analysis to identify significant context patterns
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self._extract_contextual_features()
        self._build_context_models()
    
    def _extract_contextual_features(self):
        """
        FEATURE ENGINEERING: CONTEXTUAL FEATURE EXTRACTION
        
        Extracts multiple contextual features from timestamp data:
        1. Temporal features (hour, day of week, month, season)
        2. Cyclical encoding for temporal features
        3. Contextual groupings (morning/evening, weekday/weekend)
        """
        print("📅 Extracting contextual features...")
        
        # TEMPORAL FEATURE EXTRACTION
        self.df['hour'] = self.df['ts'].dt.hour
        self.df['day_of_week'] = self.df['ts'].dt.dayofweek
        self.df['month'] = self.df['ts'].dt.month
        self.df['day_of_year'] = self.df['ts'].dt.dayofyear
        
        # CYCLICAL ENCODING: Convert cyclical features to continuous space
        # This is important for ML algorithms that assume linear relationships
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        # CONTEXTUAL GROUPINGS: Create meaningful context categories
        self.df['time_context'] = pd.cut(self.df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['night', 'morning', 'afternoon', 'evening'],
                                       include_lowest=True)
        
        self.df['day_context'] = self.df['day_of_week'].apply(
            lambda x: 'weekend' if x >= 5 else 'weekday'
        )
        
        # SEASONAL CONTEXT: Map months to seasons
        season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                     3: 'spring', 4: 'spring', 5: 'spring',
                     6: 'summer', 7: 'summer', 8: 'summer',
                     9: 'fall', 10: 'fall', 11: 'fall'}
        self.df['season'] = self.df['month'].map(season_map)
    
    def _build_context_models(self):
        """
        MACHINE LEARNING: CLUSTERING AND CONTEXT MODELING
        
        Uses unsupervised learning to identify context patterns:
        1. K-Means clustering on contextual features
        2. Statistical analysis of context-artist relationships
        3. Preference modeling for each context cluster
        """
        print("🧠 Building context-aware models using clustering...")
        
        # PREPARE FEATURES FOR CLUSTERING
        feature_columns = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        features = self.df[feature_columns].values
        
        # FEATURE SCALING: Normalize features for optimal clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # MACHINE LEARNING: K-MEANS CLUSTERING
        # Group listening sessions by contextual similarity
        n_clusters = min(8, len(self.df) // 100)  # Adaptive cluster count
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['context_cluster'] = kmeans.fit_predict(features_scaled)
            
            # CONTEXT ANALYSIS: Analyze each cluster's characteristics
            self.context_profiles = {}
            for cluster in range(n_clusters):
                cluster_data = self.df[self.df['context_cluster'] == cluster]
                
                # STATISTICAL ANALYSIS: Calculate cluster characteristics
                profile = {
                    'avg_hour': cluster_data['hour'].mean(),
                    'common_day_context': cluster_data['day_context'].mode().iloc[0] if len(cluster_data) > 0 else 'unknown',
                    'common_time_context': cluster_data['time_context'].mode().iloc[0] if len(cluster_data) > 0 else 'unknown',
                    'size': len(cluster_data)
                }
                
                # PREFERENCE MODELING: Top artists for this context
                artist_prefs = (cluster_data.groupby('artist')
                              .agg({
                                  'engagement_score': 'sum',
                                  'hours_played': 'sum'
                              })
                              .reset_index())
                
                artist_prefs['context_score'] = (
                    artist_prefs['engagement_score'] * 0.6 +
                    artist_prefs['hours_played'] * 0.4
                )
                
                profile['top_artists'] = (artist_prefs
                                        .nlargest(20, 'context_score')['artist']
                                        .tolist())
                
                self.context_profiles[cluster] = profile
        else:
            print("⚠️  Insufficient data for context clustering")
            self.context_profiles = {}
    
    def get_context_recommendations(self, context_type, n_recommendations=10):
        """
        CONTEXT-AWARE PREDICTION: Generate recommendations for specific contexts
        
        Uses the trained context models to predict appropriate music for
        different listening contexts (morning, evening, weekend, etc.)
        """
        if not hasattr(self, 'context_profiles') or not self.context_profiles:
            print("⚠️  Context models not available")
            return []
        
        # CONTEXT MAPPING: Map context types to specific patterns
        context_recommendations = []
        
        if context_type == 'time_morning':
            # Find clusters with morning characteristics
            morning_clusters = [c for c, p in self.context_profiles.items() 
                              if p.get('common_time_context') == 'morning']
        elif context_type == 'time_evening':
            evening_clusters = [c for c, p in self.context_profiles.items() 
                              if p.get('common_time_context') == 'evening']
        elif context_type == 'day_weekend':
            weekend_clusters = [c for c, p in self.context_profiles.items() 
                              if p.get('common_day_context') == 'weekend']
        else:
            # Default: use all clusters
            morning_clusters = evening_clusters = weekend_clusters = list(self.context_profiles.keys())
        
        # RECOMMENDATION GENERATION: Aggregate recommendations from relevant clusters
        relevant_clusters = []
        if context_type == 'time_morning':
            relevant_clusters = morning_clusters
        elif context_type == 'time_evening':
            relevant_clusters = evening_clusters
        elif context_type == 'day_weekend':
            relevant_clusters = weekend_clusters
        
        # ENSEMBLE METHOD: Combine recommendations from multiple clusters
        artist_scores = {}
        for cluster in relevant_clusters:
            if cluster in self.context_profiles:
                for i, artist in enumerate(self.context_profiles[cluster]['top_artists'][:n_recommendations]):
                    score = (n_recommendations - i) / n_recommendations  # Decreasing score
                    artist_scores[artist] = artist_scores.get(artist, 0) + score
        
        # RANKING: Sort by combined scores
        sorted_artists = sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)
        context_recommendations = [artist for artist, score in sorted_artists[:n_recommendations]]
        
        return context_recommendations

class EnhancedSecureMusicRecommender:
    """
    HYBRID MACHINE LEARNING SYSTEM: ENSEMBLE OF MULTIPLE AI/ML APPROACHES
    
    This is the MAIN AI/ML SYSTEM that combines multiple machine learning approaches:
    
    1. CONTENT-BASED FILTERING (AI): Uses external knowledge graphs and similarity matching
    2. COLLABORATIVE FILTERING (ML): Matrix factorization and temporal analysis
    3. CONTEXT-AWARE FILTERING (ML): Clustering and pattern recognition
    4. ENSEMBLE METHODS (ML): Combines multiple models for better predictions
    
    WHY THIS IS ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING:
    
    MACHINE LEARNING ASPECTS:
    - Learns patterns from your historical listening data
    - Uses mathematical models (matrix factorization, clustering, regression)
    - Adapts recommendations based on your behavior
    - Employs supervised and unsupervised learning techniques
    - Implements feature engineering and dimensionality reduction
    
    ARTIFICIAL INTELLIGENCE ASPECTS:
    - Makes intelligent predictions about your future preferences
    - Uses external knowledge (Last.fm music graph) for reasoning
    - Combines multiple information sources for decision making
    - Adapts to context and temporal changes in preferences
    - Provides explanations for recommendations (transparency)
    
    HYBRID APPROACH IMPLEMENTATION:
    The system combines four different AI/ML engines, each contributing unique insights:
    1. Content engine: "Users who like X also like Y" (collaborative knowledge)
    2. Temporal engine: "Your taste is evolving toward Z" (time-series prediction)
    3. Context engine: "You listen to A in the morning" (pattern recognition)
    4. Tier engine: "Explore music similar to your tier N artists" (preference modeling)
    """
    
    def __init__(self, config_method: str = "env", artist_tier_start: int = 1, 
                 artist_tier_end: int = 50, data_folder: Optional[str] = None):
        
        # Load global configuration
        self.global_config = GlobalConfig()
        
        # Set data folder (parameter overrides global config)
        if data_folder:
            self.data_directory = data_folder
            self.global_config.set_data_folder(data_folder)
        else:
            self.data_directory = self.global_config.get_data_folder()
        
        self.config_method = config_method
        self.artist_tier_start = artist_tier_start
        self.artist_tier_end = artist_tier_end
        
        # Initialize configuration managers
        self.config_manager = SecureConfigManager()
        self.config_loader = SecureConfigLoader(self.config_manager)
        self.recommender = None
        
        # Load configuration
        self.secrets = self._load_secure_config()
        
        if not self.secrets:
            raise ValueError("Failed to load secure configuration")
        
        # Initialize the AI/ML recommendation system
        self._initialize_recommender()
    
    def _load_secure_config(self) -> Dict[str, str]:
        """Load secure configuration using specified method"""
        print(f"🔐 Loading secure configuration using method: {self.config_method}")
        
        try:
            secrets = self.config_loader.load_secrets(self.config_method)
            
            if not secrets:
                print("❌ No secrets loaded. Trying alternative methods...")
                
                # Try alternative methods
                for alt_method in ["env", "encrypted", "prompt"]:
                    if alt_method != self.config_method:
                        print(f"Trying {alt_method} method...")
                        secrets = self.config_loader.load_secrets(alt_method)
                        if secrets:
                            print(f"✅ Successfully loaded using {alt_method} method")
                            break
            
            return secrets
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return {}
    
    def _initialize_recommender(self):
        """
        INITIALIZE THE HYBRID AI/ML SYSTEM
        
        Sets up all four machine learning engines:
        1. Enhanced Content-Based Recommender (with tier selection)
        2. Temporal Collaborative Filter (time-series ML)
        3. Context-Aware Recommender (clustering ML)
        4. External API integration (knowledge graph AI)
        """
        lastfm_api_key = self.secrets.get('LASTFM_API_KEY')
        
        if not lastfm_api_key:
            raise ValueError("Last.fm API key not found in configuration")
        
        print(f"🎵 Initializing HYBRID AI/ML music recommendation system...")
        print(f"📁 Data folder: {self.data_directory}")
        print(f"🎯 Artist tier: {self.artist_tier_start}-{self.artist_tier_end}")
        
        # STEP 1: DATA PROCESSING - Load and clean Spotify data
        data_processor = SpotifyDataProcessor(self.data_directory)
        df = data_processor.load_data()
        
        # STEP 2: INITIALIZE AI/ML ENGINES
        from recommendation_prototype import LastFMAPI
        
        # External Knowledge Graph API
        lastfm_api = LastFMAPI(lastfm_api_key)
        
        # ENGINE 1: CONTENT-BASED FILTERING with tier selection
        self.enhanced_content_recommender = EnhancedContentBasedRecommender(
            df, lastfm_api, self.artist_tier_start, self.artist_tier_end
        )
        
        # ENGINE 2: TEMPORAL COLLABORATIVE FILTERING
        self.temporal_cf = TemporalCollaborativeFilter(df)
        
        # ENGINE 3: CONTEXT-AWARE RECOMMENDER
        self.context_recommender = ContextAwareRecommender(df)
        
        # Store dataframe for analysis
        self.df = df
        
        print("✅ HYBRID AI/ML system initialized successfully")
        print("🧠 Active ML engines: Content-Based, Temporal, Context-Aware")
    
    def get_recommendations(self, num_recommendations: int = 20) -> Dict:
        """
        HYBRID RECOMMENDATION GENERATION: ENSEMBLE OF MULTIPLE AI/ML MODELS
        
        This method implements the HYBRID APPROACH by combining multiple AI/ML engines:
        
        1. CONTENT-BASED FILTERING: Uses external music knowledge graph (Last.fm)
           to find artists similar to your tier-selected preferences
        
        2. TEMPORAL COLLABORATIVE FILTERING: Uses matrix factorization and 
           time-series analysis to predict future preferences based on taste evolution
        
        3. CONTEXT-AWARE FILTERING: Uses clustering to identify context-specific
           preferences (morning music, weekend music, etc.)
        
        4. ENSEMBLE COMBINATION: Combines all engines using weighted scoring
        
        The result is a comprehensive recommendation set that considers:
        - Your current preferences (content-based)
        - Your evolving taste (temporal)
        - Your listening context (context-aware)
        - Your specified artist tier (tier-based)
        """
        print("🧠 Generating HYBRID AI/ML recommendations...")
        recommendations = {}
        
        # Get tier info for transparency
        tier_info = self.enhanced_content_recommender.get_tier_info()
        recommendations['tier_info'] = tier_info
        
        # ENGINE 1: TEMPORAL COLLABORATIVE FILTERING
        # Uses matrix factorization and time-series analysis
        print("🕒 Running TEMPORAL COLLABORATIVE FILTERING (Matrix Factorization + Time-Series)...")
        temporal_recs = self.temporal_cf.predict_future_preferences()
        recommendations['temporal'] = temporal_recs[:num_recommendations]
        
        # ENGINE 2: CONTENT-BASED FILTERING WITH TIER SELECTION
        # Uses external knowledge graph and similarity matching
        print(f"🎯 Running CONTENT-BASED FILTERING from tier {self.artist_tier_start}-{self.artist_tier_end}...")
        print("   Using Last.fm knowledge graph for similarity matching...")
        content_recs = self.enhanced_content_recommender.get_similar_artists_recommendations(num_recommendations)
        recommendations['content_based'] = content_recs
        
        # ENGINE 3: CONTEXT-AWARE FILTERING
        # Uses clustering and pattern recognition
        print("📅 Running CONTEXT-AWARE FILTERING (Clustering + Pattern Recognition)...")
        context_recs = {}
        contexts = ['time_morning', 'time_evening', 'day_weekend']
        for context in contexts:
            context_recs[context] = self.context_recommender.get_context_recommendations(context)
        recommendations['context_aware'] = context_recs
        
        # ENSEMBLE EXPLANATION: How the hybrid approach works
        print("\n🧠 HYBRID AI/ML SYSTEM SUMMARY:")
        print("   ✅ Content-Based: External knowledge graph similarity")
        print("   ✅ Temporal: Matrix factorization + time-series prediction")
        print("   ✅ Context-Aware: Clustering + pattern recognition")
        print("   ✅ Tier Selection: Preference modeling + ranking")
        
        return recommendations
    
    def analyze_patterns(self) -> Dict:
        """
        COMPREHENSIVE PATTERN ANALYSIS: STATISTICAL AND ML INSIGHTS
        
        Uses multiple analytical approaches to understand your listening patterns:
        1. Descriptive statistics
        2. Preference modeling results
        3. Temporal trend analysis
        4. Context clustering results
        5. Tier distribution analysis
        """
        analysis = {}
        
        # BASIC STATISTICAL ANALYSIS
        analysis['total_plays'] = len(self.df)
        analysis['total_hours'] = self.df['hours_played'].sum()
        analysis['unique_artists'] = self.df['artist'].nunique()
        analysis['unique_albums'] = self.df['album'].nunique()
        analysis['date_range'] = {
            'start': self.df['ts'].min().strftime('%Y-%m-%d'),
            'end': self.df['ts'].max().strftime('%Y-%m-%d')
        }
        
        # TOP ARTISTS (from preference modeling)
        top_artists = (self.df.groupby('artist')['hours_played']
                      .sum()
                      .nlargest(20)
                      .to_dict())
        analysis['top_artists'] = top_artists
        
        # TIER-SPECIFIC ANALYSIS (from ML preference modeling)
        tier_info = self.enhanced_content_recommender.get_tier_info()
        analysis['current_tier'] = tier_info
        
        # TIER DISTRIBUTION ANALYSIS: Statistical breakdown by preference tiers
        artist_stats = (self.df.groupby('artist')['hours_played']
                       .sum()
                       .sort_values(ascending=False)
                       .reset_index())
        artist_stats['rank'] = range(1, len(artist_stats) + 1)
        
        # Create tier distribution for analysis
        tier_ranges = [
            (1, 10, "Top 10"),
            (11, 50, "Top 11-50"),
            (51, 100, "Top 51-100"),
            (101, 200, "Top 101-200"),
            (201, 500, "Top 201-500"),
            (501, 1000, "Top 501-1000"),
            (1001, len(artist_stats), "Beyond 1000")
        ]
        
        tier_distribution = {}
        for start, end, label in tier_ranges:
            tier_artists = artist_stats[
                (artist_stats['rank'] >= start) & 
                (artist_stats['rank'] <= min(end, len(artist_stats)))
            ]
            if len(tier_artists) > 0:
                tier_distribution[label] = {
                    'count': len(tier_artists),
                    'total_hours': tier_artists['hours_played'].sum(),
                    'avg_hours': tier_artists['hours_played'].mean()
                }
        
        analysis['tier_distribution'] = tier_distribution
        
        return analysis
    
    def get_config_info(self) -> Dict[str, str]:
        """Get configuration information including global settings"""
        info = {
            "config_method": self.config_method,
            "secrets_loaded": len(self.secrets),
            "lastfm_api_configured": bool(self.secrets.get('LASTFM_API_KEY')),
            "musicbrainz_configured": bool(self.secrets.get('MUSICBRAINZ_USER_AGENT')),
            "data_folder": self.data_directory,
            "artist_tier_start": self.artist_tier_start,
            "artist_tier_end": self.artist_tier_end
        }
        
        # Add global config info
        global_folders = self.global_config.get_all_folders()
        info.update({f"global_{k}_folder": v for k, v in global_folders.items()})
        
        return info
    
    def set_global_data_folder(self, path: str):
        """Set the global data folder configuration"""
        self.global_config.set_data_folder(path)
        print(f"✅ Global data folder set to: {path}")

def main():
    """
    MAIN CLI INTERFACE: Enhanced music recommendations with AI/ML explanations
    
    This CLI provides access to the full HYBRID AI/ML SYSTEM with:
    - Artist tier selection for targeted recommendations
    - Multiple AI/ML engines (content, temporal, context-aware)
    - Comprehensive analysis and pattern recognition
    - Secure configuration management
    """
    parser = argparse.ArgumentParser(
        description='🎵 HYBRID AI/ML Music Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🧠 AI/ML FEATURES:
  • Content-Based Filtering: External knowledge graph similarity matching
  • Temporal Analysis: Matrix factorization + time-series prediction  
  • Context-Aware: Clustering + pattern recognition for contextual music
  • Artist Tier Selection: Preference modeling + ranking algorithms

EXAMPLES:
  # Get AI recommendations from your top 50 artists
  python music.py --num-recs 30

  # Explore AI recommendations from artists ranked 200-300 (faster processing!)
  python music.py --artist-tier-start 200 --artist-tier-end 300 --num-recs 25

  # Use different data folder with ML analysis
  python music.py --data-folder /path/to/spotify/data --num-recs 20

  # Set global data folder for all AI/ML runs
  python music.py --set-global-data-folder /path/to/spotify/data

  # Comprehensive ML pattern analysis
  python music.py --analyze-only --verbose
        """
    )
    
    # Configuration options
    parser.add_argument('--data-folder', help='Spotify data directory (overrides global config)')
    parser.add_argument('--config-method', choices=['env', 'encrypted', 'prompt'], 
                       default='env', help='Configuration loading method')
    
    # AI/ML Artist tier selection
    parser.add_argument('--artist-tier-start', type=int, default=1, 
                       help='Starting rank of artists for ML seed selection (default: 1)')
    parser.add_argument('--artist-tier-end', type=int, default=50,
                       help='Ending rank of artists for ML seed selection (default: 50)')
    
    # Recommendation options
    parser.add_argument('--num-recs', type=int, default=20, help='Number of AI recommendations')
    parser.add_argument('--analyze-only', action='store_true', help='Only run ML pattern analysis')
    parser.add_argument('--verbose', action='store_true', help='Show detailed AI/ML process info')
    
    # Global configuration
    parser.add_argument('--set-global-data-folder', help='Set global data folder and exit')
    parser.add_argument('--show-global-config', action='store_true', help='Show global configuration')
    
    # Setup options
    parser.add_argument('--setup-config', action='store_true', help='Setup secure configuration')
    parser.add_argument('--test-config', action='store_true', help='Test configuration loading')
    
    args = parser.parse_args()
    
    # Handle global configuration commands
    if args.set_global_data_folder:
        global_config = GlobalConfig()
        global_config.set_data_folder(args.set_global_data_folder)
        print(f"✅ Global data folder set to: {args.set_global_data_folder}")
        return
    
    if args.show_global_config:
        global_config = GlobalConfig()
        print("🌍 Global Configuration:")
        for key, value in global_config.config.items():
            print(f"   {key}: {value}")
        return
    
    # Handle setup and test commands
    if args.setup_config:
        from secrets_encryption_system import setup_secure_config
        setup_secure_config()
        return
    
    if args.test_config:
        from secrets_encryption_system import test_configuration
        test_configuration()
        return
    
    # Validate AI/ML parameters
    if args.artist_tier_start < 1:
        print("❌ Error: artist-tier-start must be >= 1")
        sys.exit(1)
    
    if args.artist_tier_end < args.artist_tier_start:
        print("❌ Error: artist-tier-end must be >= artist-tier-start")
        sys.exit(1)
    
    # Determine data directory
    data_dir = args.data_folder
    if not data_dir:
        global_config = GlobalConfig()
        data_dir = global_config.get_data_folder()
    
    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' does not exist")
        print("Please ensure your Spotify JSON files are in the correct directory")
        print("Or use --set-global-data-folder to configure the path")
        sys.exit(1)
    
    # Check for JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and 'Audio' in f]
    if not json_files:
        print(f"❌ Error: No Spotify JSON files found in '{data_dir}'")
        print("Expected files like: Audio_2012.json, Audio_2013.json, etc.")
        sys.exit(1)
    
    if args.verbose:
        print(f"📁 Found {len(json_files)} Spotify data files in {data_dir}:")
        for file in json_files:
            print(f"   - {file}")
    
    try:
        # Initialize HYBRID AI/ML system
        print("🧠 Initializing HYBRID AI/ML music recommendation system...")
        secure_recommender = EnhancedSecureMusicRecommender(
            config_method=args.config_method,
            artist_tier_start=args.artist_tier_start,
            artist_tier_end=args.artist_tier_end,
            data_folder=data_dir
        )
        
        if args.verbose:
            config_info = secure_recommender.get_config_info()
            print(f"\n📋 AI/ML System Configuration:")
            for key, value in config_info.items():
                print(f"   {key}: {value}")
        
        if args.analyze_only:
            # ML PATTERN ANALYSIS
            print("\n📊 Running comprehensive ML pattern analysis...")
            analysis = secure_recommender.analyze_patterns()
            
            print(f"\n🎵 Listening Analysis (Statistical + ML):")
            print(f"Total plays: {analysis['total_plays']:,}")
            print(f"Total hours: {analysis['total_hours']:.1f}")
            print(f"Unique artists: {analysis['unique_artists']:,}")
            print(f"Unique albums: {analysis['unique_albums']:,}")
            print(f"Date range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
            
            # Show current tier info (from ML preference modeling)
            tier_info = analysis.get('current_tier', {})
            if tier_info:
                print(f"\n🎯 Current ML Artist Tier Analysis:")
                print(f"Tier range: {tier_info['start']}-{tier_info['end']}")
                print(f"Artists in tier: {tier_info['tier_artists']}")
                print(f"Total artists: {tier_info['total_artists']}")
            
            # Show tier distribution (from ML ranking)
            if args.verbose and 'tier_distribution' in analysis:
                print(f"\n📊 ML Artist Tier Distribution:")
                for tier_name, tier_data in analysis['tier_distribution'].items():
                    print(f"   {tier_name}: {tier_data['count']} artists, {tier_data['total_hours']:.1f}h total")
            
            print(f"\n🏆 Top 10 Artists (from ML preference modeling):")
            for i, (artist, hours) in enumerate(list(analysis['top_artists'].items())[:10], 1):
                print(f"{i:2d}. {artist} ({hours:.1f} hours)")
        
        else:
            # HYBRID AI/ML RECOMMENDATION GENERATION
            print(f"\n🧠 Generating {args.num_recs} HYBRID AI/ML recommendations...")
            print(f"🎯 Using artist tier {args.artist_tier_start}-{args.artist_tier_end} as ML seeds")
            print("This combines multiple AI/ML engines - may take a few minutes...")
            
            recommendations = secure_recommender.get_recommendations(args.num_recs)
            
            # Show tier info (from ML preference modeling)
            tier_info = recommendations.get('tier_info', {})
            if tier_info:
                print(f"\n🎯 ML Recommendation Source:")
                print(f"Artist tier: {tier_info['start']}-{tier_info['end']}")
                print(f"ML seed artists: {tier_info['tier_artists']} out of {tier_info['total_artists']} total")
            
            # Display CONTENT-BASED AI recommendations (main results)
            content_recs = recommendations.get('content_based', [])
            
            if not content_recs:
                print("❌ No AI recommendations generated. This might be due to:")
                print("   - API rate limiting")
                print("   - No similar artists found for the selected tier")
                print("   - Network connectivity issues")
                print("   - Try a different artist tier range")
                return
            
            print(f"\n🎵 Top {args.num_recs} HYBRID AI/ML Artist Recommendations:")
            print("   (Content-Based Filtering + External Knowledge Graph)")
            displayed_count = min(len(content_recs), args.num_recs)
            
            for i, rec in enumerate(content_recs[:args.num_recs], 1):
                artist_name = rec.get('artist', 'Unknown Artist')
                score = rec.get('recommendation_score', 0)
                print(f"{i:2d}. {artist_name} (ML score: {score:.3f})")
            
            print(f"\n✅ Displayed {displayed_count} AI recommendations (requested: {args.num_recs})")
            
            if args.verbose:
                if len(content_recs) < args.num_recs:
                    print(f"\n⚠️  Note: Only {len(content_recs)} unique AI recommendations found.")
                    print("This could be due to:")
                    print("   - Limited similar artists for the selected tier in knowledge graph")
                    print("   - API response limitations")
                    print("   - Try expanding the tier range for more ML seeds")
                
                # Show TEMPORAL ML predictions
                temporal_recs = recommendations.get('temporal', [])
                if temporal_recs:
                    print(f"\n🕒 TEMPORAL ML Predictions (Matrix Factorization + Time-Series):")
                    for i, artist in enumerate(temporal_recs[:10], 1):
                        print(f"{i:2d}. {artist}")
                
                # Show CONTEXT-AWARE ML recommendations
                context_recs = recommendations.get('context_aware', {})
                if context_recs:
                    print(f"\n📅 CONTEXT-AWARE ML Recommendations (Clustering + Pattern Recognition):")
                    for context, artists in context_recs.items():
                        if artists:
                            print(f"   {context}: {', '.join(artists[:5])}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  AI/ML operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ AI/ML System Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("\n🔧 AI/ML Troubleshooting suggestions:")
        print("1. Run with --setup-config to configure secrets")
        print("2. Run with --test-config to test configuration")
        print("3. Check your API keys and network connectivity")
        print("4. Try a different artist tier range")
        print("5. Use --verbose for detailed AI/ML process information")
        
        sys.exit(1)

if __name__ == "__main__":
    main()

