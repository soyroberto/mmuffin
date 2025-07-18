# 🎵 Music Recommendation System - Streamlit Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Music Recommendation System web application using Streamlit. The application can be run locally for development and testing, then deployed to streamlit.io for public access.

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher (3.9+ recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: At least 2GB free space for dependencies and data
- **Network**: Internet connection for API calls and package installation

### Required Files

Ensure you have the following files in your project directory:

```
music-recommendation-system/
├── viewmusic.py                    # Main Streamlit application
├── ymusic.py                      # CLI version (for reference)
├── recommendation_prototype.py     # Core AI/ML engines
├── secrets_encryption_system.py   # Security configuration
├── requirements_streamlit.txt      # Python dependencies
├── data/                          # Your Spotify data folder
│   ├── Audio_2012.json
│   ├── Audio_2013.json
│   └── ...
└── config/                        # Configuration files (optional)
    └── global_config.json
```

## Local Development Setup

### Step 1: Environment Setup

#### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv streamlit_env

# Activate virtual environment
# On macOS/Linux:
source streamlit_env/bin/activate
# On Windows:
streamlit_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n streamlit_music python=3.9
conda activate streamlit_music
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements_streamlit.txt

# Verify Streamlit installation
streamlit --version
```

### Step 3: Configuration Setup

#### API Key Configuration

You need a Last.fm API key for the recommendation system to work:

1. **Get Last.fm API Key**:
   - Visit [Last.fm API](https://www.last.fm/api)
   - Create an account and request an API key
   - Note down your API key

2. **Configure API Key** (Choose one method):

   **Method A: Environment Variables**
   ```bash
   export LASTFM_API_KEY="your_api_key_here"
   export MUSICBRAINZ_USER_AGENT="YourMusicApp/1.0 (your.email@example.com)"
   ```

   **Method B: Configuration File**
   ```bash
   # Create config directory
   mkdir -p config
   
   # Create .env file
   cat > config/.env << 'EOF'
   LASTFM_API_KEY=your_api_key_here
   MUSICBRAINZ_USER_AGENT=YourMusicApp/1.0 (your.email@example.com)
   EOF
   ```

   **Method C: Interactive Setup** (handled by the app)
   - The app will prompt for API key on first run

#### Data Folder Setup

```bash
# Create data directory structure
mkdir -p data/spotify

# Copy your Spotify JSON files to data/spotify/
# Files should be named like: Audio_2012.json, Audio_2013.json, etc.
```

### Step 4: Run Locally

```bash
# Start the Streamlit application
streamlit run viewmusic.py

# The app will open in your browser at http://localhost:8501
```

#### Local Development Options

```bash
# Run with custom port
streamlit run viewmusic.py --server.port 8502

# Run with custom host (for network access)
streamlit run viewmusic.py --server.address 0.0.0.0

# Run in development mode (auto-reload on changes)
streamlit run viewmusic.py --server.runOnSave true
```

### Step 5: Testing Local Installation

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Configure Data**: Use sidebar to set data folder path
3. **Discover Files**: Click "Discover Data Files" to find your JSON files
4. **Select Years**: Choose which years to include
5. **Load Data**: Click "Load Data" to process your listening history
6. **Configure API**: Enter your Last.fm API key in the sidebar
7. **Test Recommendations**: Generate AI recommendations to verify everything works

## Streamlit.io Cloud Deployment

### Step 1: Prepare Repository

#### GitHub Repository Setup

1. **Create GitHub Repository**:
   ```bash
   # Initialize git repository
   git init
   
   # Add files
   git add viewmusic.py requirements_streamlit.txt
   git add recommendation_prototype.py secrets_encryption_system.py
   
   # Commit
   git commit -m "Initial commit: Music Recommendation Streamlit App"
   
   # Add remote and push
   git remote add origin https://github.com/yourusername/music-recommendations.git
   git push -u origin main
   ```

2. **Repository Structure for Deployment**:
   ```
   your-repo/
   ├── viewmusic.py                    # Main app (required)
   ├── requirements_streamlit.txt      # Dependencies (required)
   ├── recommendation_prototype.py     # Core engines
   ├── secrets_encryption_system.py   # Security system
   ├── .streamlit/                     # Streamlit config (optional)
   │   └── config.toml
   ├── README.md                       # Documentation
   └── .gitignore                      # Git ignore file
   ```

#### Important: Data Handling for Cloud Deployment

**⚠️ Security Warning**: Never commit your personal Spotify data to a public repository.

**Option A: Sample Data Approach**
```bash
# Create sample data for demonstration
mkdir -p data/spotify
echo '[{"ts":"2023-01-01T00:00:00Z","artist":"Sample Artist","track":"Sample Song","ms_played":240000}]' > data/spotify/Audio_2023.json
```

**Option B: Data Upload Feature**
- Modify the app to include file upload functionality
- Users can upload their own data files through the web interface

### Step 2: Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[global]
developmentMode = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1DB954"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Step 3: Deploy to Streamlit.io

1. **Visit Streamlit.io**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository
   - Choose branch (usually `main`)
   - Set main file path: `viewmusic.py`
   - Set requirements file: `requirements_streamlit.txt`

3. **Configure Secrets**:
   - In the Streamlit.io dashboard, go to "Secrets"
   - Add your API keys:
   ```toml
   LASTFM_API_KEY = "your_api_key_here"
   MUSICBRAINZ_USER_AGENT = "YourMusicApp/1.0 (your.email@example.com)"
   ```

4. **Deploy**:
   - Click "Deploy"
   - Wait for deployment to complete
   - Your app will be available at `https://your-app-name.streamlit.app`

### Step 4: Post-Deployment Configuration

#### Custom Domain (Optional)

If you have a custom domain:

1. **Configure DNS**:
   - Add CNAME record pointing to your Streamlit app
   - Contact Streamlit support for custom domain setup

2. **Update App Settings**:
   - Configure custom domain in Streamlit.io dashboard

#### Performance Optimization

1. **Resource Limits**:
   - Streamlit.io has resource limits (1GB RAM, 1 CPU core)
   - Optimize data processing for cloud constraints
   - Consider data sampling for large datasets

2. **Caching Strategy**:
   ```python
   @st.cache_data
   def load_data(data_folder, years):
       # Cached data loading
       pass
   
   @st.cache_resource
   def initialize_recommender(api_key):
       # Cached recommender initialization
       pass
   ```

## Advanced Deployment Options

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "viewmusic.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
# Build Docker image
docker build -t music-recommendations .

# Run container
docker run -p 8501:8501 music-recommendations
```

### Heroku Deployment

Create `Procfile`:

```
web: sh setup.sh && streamlit run viewmusic.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:

```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

Deploy:

```bash
# Install Heroku CLI and login
heroku create your-music-app
git push heroku main
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` for custom modules

**Solution**:
```python
# Add to top of viewmusic.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

#### 2. API Key Issues

**Problem**: API calls failing

**Solutions**:
- Verify API key is correct
- Check API rate limits
- Ensure network connectivity
- Test API key with curl:
  ```bash
  curl "http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar&artist=cher&api_key=YOUR_API_KEY&format=json"
  ```

#### 3. Memory Issues

**Problem**: App crashes due to memory limits

**Solutions**:
- Implement data sampling for large datasets
- Use streaming data processing
- Optimize pandas operations
- Clear unused variables

#### 4. File Upload Issues

**Problem**: Cannot find data files

**Solutions**:
- Check file paths are absolute
- Verify file permissions
- Ensure JSON files are valid format
- Use file upload widget for cloud deployment

#### 5. Streamlit.io Deployment Issues

**Problem**: Deployment fails

**Solutions**:
- Check requirements.txt format
- Verify all imports are available
- Review deployment logs
- Test locally first
- Check file size limits (25MB per file)

### Performance Optimization

#### Data Processing Optimization

```python
# Use efficient pandas operations
@st.cache_data
def process_large_dataset(df):
    # Vectorized operations instead of loops
    df['engagement_score'] = df['ms_played'] / (3.5 * 60 * 1000)
    return df

# Memory-efficient data loading
def load_data_chunked(files):
    chunks = []
    for file in files:
        chunk = pd.read_json(file, lines=True, chunksize=10000)
        chunks.extend(chunk)
    return pd.concat(chunks, ignore_index=True)
```

#### UI Performance

```python
# Use session state for expensive operations
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = process_data()

# Lazy loading for large datasets
@st.cache_data
def get_artist_subset(start, end):
    return artists[start:end]
```

## Security Considerations

### API Key Security

1. **Never commit API keys to version control**
2. **Use Streamlit secrets for cloud deployment**
3. **Implement key rotation procedures**
4. **Monitor API usage for anomalies**

### Data Privacy

1. **Personal data should not be committed to public repos**
2. **Implement data upload functionality for user data**
3. **Consider data anonymization for demo purposes**
4. **Comply with data protection regulations**

### Application Security

```python
# Input validation
def validate_api_key(key):
    if not key or len(key) < 10:
        return False
    return True

# Error handling
try:
    recommendations = get_recommendations()
except Exception as e:
    st.error("An error occurred. Please try again.")
    # Log error securely without exposing sensitive info
```

## Monitoring and Maintenance

### Application Monitoring

1. **Monitor app performance metrics**
2. **Track user engagement and errors**
3. **Set up alerts for downtime**
4. **Regular dependency updates**

### Maintenance Tasks

```bash
# Regular maintenance checklist
# 1. Update dependencies
pip list --outdated
pip install --upgrade package_name

# 2. Check security vulnerabilities
pip audit

# 3. Update API keys if needed
# 4. Monitor resource usage
# 5. Review and update documentation
```

## Conclusion

This deployment guide provides comprehensive instructions for running the Music Recommendation System both locally and on Streamlit.io. The application combines powerful AI/ML algorithms with an intuitive web interface, making it accessible to users while maintaining the sophisticated functionality of the CLI version.

For additional support or questions, refer to the project repository or contact the development team.

