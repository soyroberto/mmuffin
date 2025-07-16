#!/usr/bin/env python3
"""
Secure Music Recommendation System
Integrates with the secrets encryption system for secure API key management

This version automatically loads encrypted configuration and provides
secure handling of sensitive information.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import argparse

# Import the secure configuration system
from secrets_encryption_system import SecureConfigManager, SecureConfigLoader

# Import the fixed recommendation system
from fixed_recommendation_prototype import HybridMusicRecommender

class SecureMusicRecommender:
    """
    Secure wrapper for the music recommendation system
    Handles encrypted configuration and secure API key management
    """
    
    def __init__(self, data_directory: str, config_method: str = "env"):
        self.data_directory = data_directory
        self.config_method = config_method
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
        """
        Load secure configuration using specified method
        """
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
        Initialize the music recommendation system with secure configuration
        """
        lastfm_api_key = self.secrets.get('LASTFM_API_KEY')
        
        if not lastfm_api_key:
            raise ValueError("Last.fm API key not found in configuration")
        
        print("🎵 Initializing music recommendation system...")
        self.recommender = HybridMusicRecommender(
            data_directory=self.data_directory,
            lastfm_api_key=lastfm_api_key
        )
        
        print("✅ Secure music recommender initialized successfully")
    
    def get_recommendations(self, num_recommendations: int = 20) -> Dict:
        """
        Get music recommendations securely
        """
        if not self.recommender:
            raise ValueError("Recommender not initialized")
        
        return self.recommender.get_comprehensive_recommendations(num_recommendations)
    
    def analyze_patterns(self) -> Dict:
        """
        Analyze listening patterns securely
        """
        if not self.recommender:
            raise ValueError("Recommender not initialized")
        
        return self.recommender.analyze_listening_patterns()
    
    def get_config_info(self) -> Dict[str, str]:
        """
        Get configuration information (without exposing secrets)
        """
        info = {
            "config_method": self.config_method,
            "secrets_loaded": len(self.secrets),
            "lastfm_api_configured": bool(self.secrets.get('LASTFM_API_KEY')),
            "musicbrainz_configured": bool(self.secrets.get('MUSICBRAINZ_USER_AGENT'))
        }
        
        return info

def main():
    """
    Secure CLI for music recommendations
    """
    parser = argparse.ArgumentParser(description='Secure Music Recommendation System')
    parser.add_argument('--data-dir', default='/Users/roberto/OneDrive/Azure/Spotify/spai', help='Spotify data directory')
    parser.add_argument('--num-recs', type=int, default=20, help='Number of recommendations')
    parser.add_argument('--config-method', choices=['env', 'encrypted', 'prompt'], 
                       default='env', help='Configuration loading method')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze patterns')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--setup-config', action='store_true', help='Setup secure configuration')
    parser.add_argument('--test-config', action='store_true', help='Test configuration loading')
    
    args = parser.parse_args()
    
    # Handle setup and test commands
    if args.setup_config:
        from secrets_encryption_system import setup_secure_config
        setup_secure_config()
        return
    
    if args.test_config:
        from secrets_encryption_system import test_configuration
        test_configuration()
        return
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' does not exist")
        print("Please ensure your Spotify JSON files are in the correct directory")
        sys.exit(1)
    
    # Check for JSON files
    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json') and 'Audio' in f]
    if not json_files:
        print(f"❌ Error: No Spotify JSON files found in '{args.data_dir}'")
        print("Expected files like: Audio_2012.json, Audio_2013.json, etc.")
        sys.exit(1)
    
    if args.verbose:
        print(f"📁 Found {len(json_files)} Spotify data files:")
        for file in json_files:
            print(f"   - {file}")
    
    try:
        # Initialize secure recommender
        print("🔐 Initializing secure music recommendation system...")
        secure_recommender = SecureMusicRecommender(
            data_directory=args.data_dir,
            config_method=args.config_method
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
            import random
            random_num= random.randint(10, 33)
            print(f"\n🏆 Top  Artists by Hours:")
            #for i, (artist, hours) in enumerate(list(analysis['top_artists'].items())[:10], 1):
            for i, (artist, hours) in enumerate(list(analysis['top_artists'].items())[:random_num], 1):
                print(f"{i:2d}. {artist} ({hours:.1f} hours)")
        
        else:
            # Generate recommendations
            print(f"\n🎵 Generating {args.num_recs} secure recommendations...")
            print("This may take a few minutes due to API calls...")
            
            recommendations = secure_recommender.get_recommendations(args.num_recs)
            
            # Display content-based recommendations
            content_recs = recommendations.get('content_based', [])
            
            if not content_recs:
                print("❌ No recommendations generated. This might be due to:")
                print("   - API rate limiting")
                print("   - No similar artists found")
                print("   - Network connectivity issues")
                print("   - Configuration problems")
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
                    print("   - Limited similar artists in Last.fm database")
                    print("   - Filtering out artists already in your library")
                    print("   - API response limitations")
                
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
        print("4. Use --verbose for detailed error information")
        
        sys.exit(1)

if __name__ == "__main__":
    main()

