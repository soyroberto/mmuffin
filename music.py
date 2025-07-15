#!/usr/bin/env python3

import argparse
#from recommendation_prototype import HybridMusicRecommender
from main import HybridMusicRecommender

def main():
    parser = argparse.ArgumentParser(description='Music Recommendation System')
    parser.add_argument('--data-dir', default='/Users/roberto/OneDrive/Azure/Spotify/spai', help='Spotify data directory')
    parser.add_argument('--num-recs', type=int, default=20, help='Number of recommendations')
    parser.add_argument('--api-key', help='Last.fm API key')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze patterns')
    
    args = parser.parse_args()
    
    # Initialize recommender
    recommender = HybridMusicRecommender(
        data_directory=args.data_dir,
        lastfm_api_key=args.api_key
    )
    
    if args.analyze_only:
        # Just analyze patterns
        analysis = recommender.analyze_listening_patterns()
        print(f"Listening Analysis:")
        print(f"Total plays: {analysis['total_plays']}")
        print(f"Total hours: {analysis['total_hours']:.1f}")
        print(f"Unique artists: {analysis['unique_artists']}")
        print(f"Date range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
    else:
        # Generate recommendations
        recommendations = recommender.get_comprehensive_recommendations(args.num_recs)
        
        print(f"\nTop {args.num_recs} Artist Recommendations:")
        for i, rec in enumerate(recommendations['content_based'][:args.num_recs], 1):
            print(f"{i:2d}. {rec['artist']} (score: {rec['recommendation_score']:.3f})")

if __name__ == "__main__":
    main()