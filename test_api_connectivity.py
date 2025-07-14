#!/usr/bin/env python3

import requests
import time
from config import LASTFM_API_KEY, MUSICBRAINZ_USER_AGENT

def test_lastfm_api():
    """Test Last.fm API connectivity"""
    if not LASTFM_API_KEY or LASTFM_API_KEY == "your_lastfm_api_key_here":
        print("❌ Last.fm API key not configured")
        return False
    
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        'method': 'artist.getinfo',
        'artist': 'The Beatles',
        'api_key': LASTFM_API_KEY,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'artist' in data:
            print("✅ Last.fm API test passed!")
            return True
        else:
            print("❌ Last.fm API returned unexpected response")
            return False
    except Exception as e:
        print(f"❌ Last.fm API test failed: {e}")
        return False

def test_musicbrainz_api():
    """Test MusicBrainz API connectivity"""
    url = "https://musicbrainz.org/ws/2/artist/"
    params = {
        'query': 'artist:"The Beatles"',
        'fmt': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': MUSICBRAINZ_USER_AGENT
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'artists' in data and data['artists']:
            print("✅ MusicBrainz API test passed!")
            return True
        else:
            print("❌ MusicBrainz API returned unexpected response")
            return False
    except Exception as e:
        print(f"❌ MusicBrainz API test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing API connectivity...")
    
    lastfm_ok = test_lastfm_api()
    time.sleep(1)  # Rate limiting
    musicbrainz_ok = test_musicbrainz_api()
    
    if lastfm_ok and musicbrainz_ok:
        print("✅ All API tests passed!")
    else:
        print("⚠️  Some API tests failed. Check your configuration.")