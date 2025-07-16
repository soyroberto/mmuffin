#!/usr/bin/env python3
"""
Enhanced Secure Music Recommendation System
New Features:
- Artist tier selection (e.g., artists ranked 200-300)
- Global data folder configuration
- Improved performance for different artist ranges
- Enhanced CLI with more options

This version allows you to explore recommendations from different tiers of your
listening history, making it faster to get recommendations from less-played artists.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse

# Import the secure configuration system
from secrets_encryption_system import SecureConfigManager, SecureConfigLoader

# Import the fixed recommendation system
from fixed_recommendation_prototype import HybridMusicRecommender, SpotifyDataProcessor, ContentBasedRecommender

class GlobalConfig:
    """
    Global configuration manager for the music recommendation system
    Handles data folder paths and other global settings
    """
    
    def __init__(self, config_file: str = "config/global_config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load global configuration from file"""
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
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load global config: {e}")
                print("Using default configuration")
        
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
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value
    
    def get_data_folder(self) -> str:
        """Get the configured data folder path"""
        return self.config["data_folder"]
    
    def set_data_folder(self, path: str):
        """Set the data folder path"""
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
    Enhanced content-based recommender with artist tier selection
    """
    
    def __init__(self, df, lastfm_api=None, artist_tier_start=1, artist_tier_end=50):
        super().__init__(df, lastfm_api)
        self.artist_tier_start = artist_tier_start
        self.artist_tier_end = artist_tier_end
        self._rebuild_user_profile_with_tiers()
    
    def _rebuild_user_profile_with_tiers(self):
        """Rebuild user profile with specified artist tiers"""
        print(f"🎯 Building user profile with artist tiers {self.artist_tier_start}-{self.artist_tier_end}")
        
        # Calculate artist preferences (same as before)
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
        
        # Sort by preference score
        artist_stats = artist_stats.sort_values('preference_score', ascending=False).reset_index(drop=True)
        
        # Add ranking
        artist_stats['rank'] = range(1, len(artist_stats) + 1)
        
        # Select artists within the specified tier range
        tier_mask = (
            (artist_stats['rank'] >= self.artist_tier_start) & 
            (artist_stats['rank'] <= self.artist_tier_end)
        )
        tier_artists = artist_stats[tier_mask]
        
        print(f"📊 Total artists in your library: {len(artist_stats)}")
        print(f"🎯 Artists in tier {self.artist_tier_start}-{self.artist_tier_end}: {len(tier_artists)}")
        
        if len(tier_artists) == 0:
            print(f"⚠️  Warning: No artists found in tier {self.artist_tier_start}-{self.artist_tier_end}")
            print(f"Available range: 1-{len(artist_stats)}")
            # Fall back to top artists
            tier_artists = artist_stats.head(min(50, len(artist_stats)))
            print(f"Falling back to top {len(tier_artists)} artists")
        
        # Update user profile with tier-specific artists
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
        
        # Show some examples of tier artists
        print(f"🎵 Sample artists in this tier:")
        for i, (_, row) in enumerate(tier_artists.head(5).iterrows()):
            print(f"   #{row['rank']}: {row['artist']} ({row['total_hours']:.1f}h, score: {row['preference_score']:.2f})")
    
    def get_tier_info(self) -> Dict:
        """Get information about the current artist tier"""
        return self.user_profile.get('tier_info', {})

class EnhancedSecureMusicRecommender:
    """
    Enhanced secure wrapper for the music recommendation system
    Includes artist tier selection and global configuration
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
        
        # Initialize the recommendation system
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
        """Initialize the music recommendation system with enhanced features"""
        lastfm_api_key = self.secrets.get('LASTFM_API_KEY')
        
        if not lastfm_api_key:
            raise ValueError("Last.fm API key not found in configuration")
        
        print(f"🎵 Initializing enhanced music recommendation system...")
        print(f"📁 Data folder: {self.data_directory}")
        print(f"🎯 Artist tier: {self.artist_tier_start}-{self.artist_tier_end}")
        
        # Initialize data processor
        data_processor = SpotifyDataProcessor(self.data_directory)
        df = data_processor.load_data()
        
        # Initialize enhanced content recommender with tier selection
        from fixed_recommendation_prototype import LastFMAPI, TemporalCollaborativeFilter, ContextAwareRecommender
        
        lastfm_api = LastFMAPI(lastfm_api_key)
        
        self.enhanced_content_recommender = EnhancedContentBasedRecommender(
            df, lastfm_api, self.artist_tier_start, self.artist_tier_end
        )
        
        # Initialize other components
        self.temporal_cf = TemporalCollaborativeFilter(df)
        self.context_recommender = ContextAwareRecommender(df)
        
        # Store dataframe for analysis
        self.df = df
        
        print("✅ Enhanced secure music recommender initialized successfully")
    
    def get_recommendations(self, num_recommendations: int = 20) -> Dict:
        """Get music recommendations with tier-specific seed artists"""
        recommendations = {}
        
        # Get tier info
        tier_info = self.enhanced_content_recommender.get_tier_info()
        recommendations['tier_info'] = tier_info
        
        # Temporal collaborative filtering (uses all data)
        print("🕒 Generating temporal recommendations...")
        temporal_recs = self.temporal_cf.predict_future_preferences()
        recommendations['temporal'] = temporal_recs[:num_recommendations]
        
        # Enhanced content-based recommendations (uses tier-specific artists)
        print(f"🎯 Generating content-based recommendations from tier {self.artist_tier_start}-{self.artist_tier_end}...")
        content_recs = self.enhanced_content_recommender.get_similar_artists_recommendations(num_recommendations)
        recommendations['content_based'] = content_recs
        
        # Context-aware recommendations
        print("📅 Generating context-aware recommendations...")
        context_recs = {}
        for context in ['time_morning', 'time_evening', 'day_saturday']:
            context_recs[context] = self.context_recommender.get_context_recommendations(context)
        recommendations['context_aware'] = context_recs
        
        return recommendations
    
    def analyze_patterns(self) -> Dict:
        """Analyze listening patterns with tier information"""
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
                      .nlargest(20)
                      .to_dict())
        analysis['top_artists'] = top_artists
        
        # Tier-specific analysis
        tier_info = self.enhanced_content_recommender.get_tier_info()
        analysis['current_tier'] = tier_info
        
        # Artist distribution by tiers
        artist_stats = (self.df.groupby('artist')['hours_played']
                       .sum()
                       .sort_values(ascending=False)
                       .reset_index())
        artist_stats['rank'] = range(1, len(artist_stats) + 1)
        
        # Create tier distribution
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
    """Enhanced CLI for music recommendations with tier selection"""
    parser = argparse.ArgumentParser(
        description='Enhanced Secure Music Recommendation System with Artist Tier Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get recommendations from your top 50 artists
  python enhanced_secure_music_recommender.py --num-recs 30

  # Get recommendations from artists ranked 200-300 in your library
  python enhanced_secure_music_recommender.py --artist-tier-start 200 --artist-tier-end 300 --num-recs 25

  # Use a different data folder
  python enhanced_secure_music_recommender.py --data-folder /path/to/spotify/data --num-recs 20

  # Set global data folder for all future runs
  python enhanced_secure_music_recommender.py --set-global-data-folder /path/to/spotify/data

  # Analyze listening patterns with tier distribution
  python enhanced_secure_music_recommender.py --analyze-only --verbose
        """
    )
    
    # Configuration options
    parser.add_argument('--data-folder', help='Spotify data directory (overrides global config)')
    parser.add_argument('--config-method', choices=['env', 'encrypted', 'prompt'], 
                       default='env', help='Configuration loading method')
    
    # Artist tier selection
    parser.add_argument('--artist-tier-start', type=int, default=1, 
                       help='Starting rank of artists to use as seeds (default: 1)')
    parser.add_argument('--artist-tier-end', type=int, default=50,
                       help='Ending rank of artists to use as seeds (default: 50)')
    
    # Recommendation options
    parser.add_argument('--num-recs', type=int, default=20, help='Number of recommendations')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze patterns')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
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
    
    # Validate artist tier parameters
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
        # Initialize enhanced secure recommender
        print("🔐 Initializing enhanced secure music recommendation system...")
        secure_recommender = EnhancedSecureMusicRecommender(
            config_method=args.config_method,
            artist_tier_start=args.artist_tier_start,
            artist_tier_end=args.artist_tier_end,
            data_folder=data_dir
        )
        
        if args.verbose:
            config_info = secure_recommender.get_config_info()
            print(f"\n📋 Configuration Info:")
            for key, value in config_info.items():
                print(f"   {key}: {value}")
        
        if args.analyze_only:
            # Analyze patterns only
            print("\n📊 Analyzing listening patterns...")
            analysis = secure_recommender.analyze_patterns()
            
            print(f"\n🎵 Listening Analysis:")
            print(f"Total plays: {analysis['total_plays']:,}")
            print(f"Total hours: {analysis['total_hours']:.1f}")
            print(f"Unique artists: {analysis['unique_artists']:,}")
            print(f"Unique albums: {analysis['unique_albums']:,}")
            print(f"Date range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
            
            # Show current tier info
            tier_info = analysis.get('current_tier', {})
            if tier_info:
                print(f"\n🎯 Current Artist Tier:")
                print(f"Tier range: {tier_info['start']}-{tier_info['end']}")
                print(f"Artists in tier: {tier_info['tier_artists']}")
                print(f"Total artists: {tier_info['total_artists']}")
            
            # Show tier distribution
            if args.verbose and 'tier_distribution' in analysis:
                print(f"\n📊 Artist Tier Distribution:")
                for tier_name, tier_data in analysis['tier_distribution'].items():
                    print(f"   {tier_name}: {tier_data['count']} artists, {tier_data['total_hours']:.1f}h total")
            
            print(f"\n🏆 Top 10 Artists by Hours:")
            for i, (artist, hours) in enumerate(list(analysis['top_artists'].items())[:10], 1):
                print(f"{i:2d}. {artist} ({hours:.1f} hours)")
        
        else:
            # Generate recommendations
            print(f"\n🎵 Generating {args.num_recs} recommendations from artist tier {args.artist_tier_start}-{args.artist_tier_end}...")
            print("This may take a few minutes due to API calls...")
            
            recommendations = secure_recommender.get_recommendations(args.num_recs)
            
            # Show tier info
            tier_info = recommendations.get('tier_info', {})
            if tier_info:
                print(f"\n🎯 Recommendation Source:")
                print(f"Artist tier: {tier_info['start']}-{tier_info['end']}")
                print(f"Seed artists: {tier_info['tier_artists']} out of {tier_info['total_artists']} total")
            
            # Display content-based recommendations
            content_recs = recommendations.get('content_based', [])
            
            if not content_recs:
                print("❌ No recommendations generated. This might be due to:")
                print("   - API rate limiting")
                print("   - No similar artists found for the selected tier")
                print("   - Network connectivity issues")
                print("   - Try a different artist tier range")
                return
            
            print(f"\n🎵 Top {args.num_recs} Artist Recommendations:")
            displayed_count = min(len(content_recs), args.num_recs)
            
            for i, rec in enumerate(content_recs[:args.num_recs], 1):
                artist_name = rec.get('artist', 'Unknown Artist')
                score = rec.get('recommendation_score', 0)
                print(f"{i:2d}. {artist_name} (score: {score:.3f})")
            
            print(f"\n✅ Displayed {displayed_count} recommendations (requested: {args.num_recs})")
            
            if args.verbose:
                if len(content_recs) < args.num_recs:
                    print(f"\n⚠️  Note: Only {len(content_recs)} unique recommendations found.")
                    print("This could be due to:")
                    print("   - Limited similar artists for the selected tier")
                    print("   - API response limitations")
                    print("   - Try expanding the tier range")
                
                # Show temporal recommendations
                temporal_recs = recommendations.get('temporal', [])
                if temporal_recs:
                    print(f"\n🕒 Temporal Predictions (based on taste evolution):")
                    for i, artist in enumerate(temporal_recs[:10], 1):
                        print(f"{i:2d}. {artist}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Run with --setup-config to configure secrets")
        print("2. Run with --test-config to test configuration")
        print("3. Check your API keys and network connectivity")
        print("4. Try a different artist tier range")
        print("5. Use --verbose for detailed error information")
        
        sys.exit(1)

if __name__ == "__main__":
    # Import numpy here to avoid issues with the enhanced recommender
    import numpy as np
    main()

