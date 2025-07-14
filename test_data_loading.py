#!/usr/bin/env python3

import os
import json
import pandas as pd
from pathlib import Path

def test_data_loading():
    """Test loading Spotify data files"""
    data_dir = Path("/Users/roberto/OneDrive/Azure/Spotify/spai").expanduser()
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return False
    
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {data_dir}")
        return False
    
    print(f"Found {len(json_files)} JSON files")
    
    total_records = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records = len(data)
                total_records += records
                print(f"{json_file.name}: {records} records")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            return False
    
    print(f"Total records: {total_records}")
    
    # Test pandas loading
    try:
        all_data = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        df = pd.DataFrame(all_data)
        print(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
        print(f"Unique artists: {df['master_metadata_album_artist_name'].nunique()}")
        
        return True
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("✅ Data loading test passed!")
    else:
        print("❌ Data loading test failed!")