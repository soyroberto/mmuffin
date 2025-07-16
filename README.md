# mmuffin
# 🎵 Personal Music Recommendation System

A comprehensive, secure music recommendation system that analyzes your Spotify listening history and provides personalized recommendations using machine learning and external music APIs. Built with privacy and security in mind, this system processes your data locally and encrypts all sensitive configuration.

## 🌟 Features

### 🎯 **Advanced Recommendation Engine**
- **Hybrid Approach**: Combines collaborative filtering, content-based filtering, and temporal analysis
- **Artist Tier Selection**: Get recommendations from specific artist popularity ranges (e.g., artists ranked 200-300)
- **Temporal Analysis**: Tracks how your musical taste evolves over time
- **Context-Aware**: Considers listening patterns by time of day, day of week, and seasons
- **Multi-Modal**: Integrates Last.fm API for enhanced music metadata and similarity

### 🔐 **Security & Privacy**
- **Encrypted Configuration**: Multiple encryption methods for API keys and sensitive data
- **Local Processing**: All analysis happens on your machine
- **Git-Safe**: Comprehensive protection against accidental secret commits
- **Multiple Auth Methods**: Environment variables, encrypted files, or interactive prompts

### ⚙️ **Flexible Configuration**
- **Global Settings**: Centralized configuration for data folders and system parameters
- **Tier-Based Analysis**: Focus on different segments of your music library
- **Customizable Algorithms**: Adjustable weights and parameters for recommendation engines
- **Caching System**: Intelligent caching to reduce API calls and improve performance

## 🧠 How It Works

### **Data Processing Pipeline**

The system processes your Spotify data through several sophisticated stages to generate personalized recommendations that go beyond simple popularity-based suggestions.

**Stage 1: Data Ingestion and Cleaning**
The system begins by loading your Spotify Extended Streaming History JSON files, which contain detailed information about every track you've played. This data includes timestamps, track names, artist names, album information, and crucially, the actual listening duration (ms_played). The system performs comprehensive data cleaning, filtering out incomplete entries, removing tracks with '(Blank)' values, and calculating engagement scores based on the percentage of each track actually listened to.

**Stage 2: Feature Engineering**
Raw listening data is transformed into meaningful features that capture your musical preferences. The system calculates engagement scores by comparing actual listening time to track duration, identifying songs you skip versus those you play completely. Temporal features are extracted to understand when you listen to different types of music, and artist preference scores are computed using a weighted combination of total listening time, average engagement, play frequency, and recency of listening.

**Stage 3: Multi-Modal Recommendation Generation**
The system employs four distinct recommendation engines that work together to provide comprehensive suggestions:

**Temporal Collaborative Filtering** analyzes how your musical taste has evolved over time, identifying patterns in your listening history to predict what you might enjoy next based on your taste trajectory. This engine uses matrix factorization techniques to model the relationship between time periods and artist preferences, allowing it to suggest artists that align with your evolving musical journey.

**Content-Based Filtering with Artist Tiers** leverages the Last.fm API to find artists similar to those in your specified preference tier. Rather than always using your most-played artists as seeds, you can specify a tier range (e.g., artists ranked 200-300 in your library) to discover recommendations based on your less-mainstream preferences. This approach prevents the system from only suggesting popular artists and helps you explore music similar to your deeper cuts.

**Context-Aware Recommendations** analyze your listening patterns across different contexts such as time of day, day of the week, and seasonal patterns. The system identifies that you might prefer energetic music in the morning, ambient music in the evening, or different genres on weekends versus weekdays, then generates recommendations that match these contextual preferences.

**Hybrid Ensemble** combines all recommendation engines using a sophisticated scoring system that weights each engine's suggestions based on confidence scores, user preference patterns, and recommendation diversity to provide a final ranked list of artist suggestions.

### **Security Architecture**

The system implements multiple layers of security to protect your personal data and API credentials while maintaining ease of use.

**Encryption Layer** provides three different methods for securing your API keys and sensitive configuration. Environment variables offer the simplest approach for local development, storing secrets in .env files that are automatically excluded from version control. Fernet encryption provides symmetric encryption with password-based key derivation, allowing you to store encrypted configuration files that can be safely committed to public repositories. AES-256-CBC encryption offers maximum security for highly sensitive environments, using advanced encryption standards with secure key derivation functions.

**Configuration Management** handles the complex task of loading secrets from multiple sources with a clear priority hierarchy. The system first checks for environment variables, then looks for .env files, attempts to decrypt configuration files, and finally falls back to interactive prompts. This approach ensures that the system works in development, testing, and production environments while maintaining security best practices.

**Git Safety** includes comprehensive .gitignore configurations and automated checks to prevent accidental commits of sensitive data. The system generates appropriate .gitignore entries for all sensitive files and provides templates for sharing configuration without exposing secrets.

## 📋 Prerequisites

### **System Requirements**

**Operating System**: The system is designed to work on macOS, Linux, and Windows, with specific optimizations for ARM-based macOS systems (Apple Silicon M1/M2/M3/M4). The system has been tested extensively on macOS Monterey and later, Ubuntu 20.04 and later, and Windows 10/11 with WSL2.

**Python Environment**: Python 3.8 or higher is required, with Python 3.11 recommended for optimal performance. The system uses modern Python features and type hints that require recent Python versions. Virtual environment usage is strongly recommended to avoid dependency conflicts.

**Memory and Storage**: The system requires at least 4GB of RAM for processing large Spotify datasets, with 8GB recommended for datasets spanning multiple years. Storage requirements depend on your data size, but typically 1-2GB of free space is sufficient for the application, cache, and temporary files.

**Network Connectivity**: Internet access is required for API calls to Last.fm and MusicBrainz services. The system includes intelligent rate limiting and caching to minimize network usage, but initial setup and recommendation generation require active internet connectivity.

### **Data Requirements**

**Spotify Extended Streaming History**: You need to request and download your Extended Streaming History from Spotify, which provides detailed listening data including exact play times and engagement metrics. This data is more comprehensive than the standard account data and is essential for accurate recommendation generation.

To obtain your Extended Streaming History, log into your Spotify account, go to Privacy Settings, and request "Extended streaming history" under the Data Download section. This process typically takes 5-30 days for Spotify to prepare your data. The resulting download will contain JSON files named like "Audio_2012.json", "Audio_2013.json", etc., containing your listening history for each year.

**API Access**: You'll need a free Last.fm API key for music metadata and similarity information. Last.fm provides comprehensive music data including artist relationships, tags, and similarity scores that enhance the recommendation quality. Registration is free and immediate at https://www.last.fm/api/account/create.

### **Technical Dependencies**

**Core Libraries**: The system relies on several key Python libraries for different aspects of functionality. Pandas and NumPy handle data processing and numerical computations, with specific optimizations for large datasets. Scikit-learn provides machine learning algorithms for collaborative filtering and clustering. The implicit library offers specialized collaborative filtering algorithms optimized for implicit feedback data like listening history.

**Security Libraries**: Cryptography library provides encryption and decryption capabilities for secure configuration management. Python-dotenv handles environment variable loading from .env files. These libraries ensure that your API keys and sensitive data remain protected throughout the system lifecycle.

**API and Network Libraries**: Requests library handles HTTP communications with external APIs, including proper error handling and retry logic. BeautifulSoup assists with parsing API responses and handling malformed data. The system includes comprehensive error handling for network issues and API rate limiting.

**Visualization Libraries**: Matplotlib, Seaborn, and Plotly provide data visualization capabilities for analyzing your listening patterns and recommendation results. These libraries are optional but recommended for gaining insights into your musical preferences and system performance.

## 🚀 Quick Start

### **Installation Process**

The installation process is designed to be straightforward while ensuring all security measures are properly configured from the beginning.

**Step 1: Repository Setup**
Clone or download the repository to your local machine and navigate to the project directory. If you're setting up the system from individual files, ensure all core components are in the same directory: the enhanced secure music recommender, secrets encryption system, and fixed recommendation prototype.

**Step 2: Dependency Installation**
Install the required Python packages using the provided requirements file or individual package installation. The system includes specific version requirements to ensure compatibility and security. For ARM-based macOS systems, the installation process includes optimizations for Apple Silicon processors.

```bash
# Install core dependencies
pip install cryptography>=41.0.0 python-dotenv>=1.0.0
pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0
pip install requests>=2.31.0 implicit>=0.7.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.15.0
```

**Step 3: Automated Setup**
Run the automated setup script to configure directory structure, security settings, and initial configuration. This script creates necessary directories, sets up .gitignore files, and guides you through the security configuration process.

```bash
python setup_secure_system.py
```

**Step 4: Security Configuration**
Configure your API keys and sensitive settings using the interactive setup process. The system will guide you through obtaining necessary API keys and choosing the appropriate security method for your environment.

```bash
python music.py --setup-config
```

### **Data Preparation**

**Spotify Data Organization**
Create a data directory structure and place your Spotify Extended Streaming History JSON files in the appropriate location. The system expects files named with the pattern "Audio_YYYY.json" where YYYY represents the year.

```bash
mkdir -p data/spotify
# Copy your Spotify JSON files to data/spotify/
# Files should be named: Audio_2012.json, Audio_2013.json, etc.
```

**Global Configuration**
Set up global configuration for data folders and system parameters that will be used across all future runs of the system.

```bash
# Set global data folder
python music.py --set-global-data-folder data/spotify

# View current global configuration
python music.py --show-global-config
```

### **First Run**

**Basic Analysis**
Start with a basic analysis of your listening patterns to verify that the system is working correctly and to understand your musical profile.

```bash
# Analyze your listening patterns
python music.py --analyze-only --verbose
```

**Generate Recommendations**
Create your first set of recommendations using the default settings, which will use your top 50 artists as seeds for finding similar music.

```bash
# Generate 20 recommendations from your top artists
python music.py --num-recs 20
```

**Explore Artist Tiers**
Experiment with different artist tiers to discover music based on different segments of your listening history.

```bash
# Get recommendations from artists ranked 100-200 in your library
python music.py --artist-tier-start 100 --artist-tier-end 200 --num-recs 25
```



## 📖 Detailed Usage

### **Artist Tier Selection**

One of the most powerful features of this system is the ability to generate recommendations based on specific tiers of your music library, rather than always using your most-played artists as seeds.

**Understanding Artist Tiers**
Your music library is automatically ranked by preference score, which combines listening time, engagement rate, play frequency, and recency. Artist tier selection allows you to focus recommendation generation on specific segments of this ranked list, enabling discovery of music similar to your less mainstream preferences.

**Tier Selection Strategies**
Different tier ranges serve different discovery purposes and can reveal different aspects of your musical taste. Using your top-tier artists (ranks 1-50) generates recommendations similar to your most-played music, which tends to be safe but potentially predictable. Mid-tier artists (ranks 100-300) often represent your more adventurous listening and can lead to more diverse recommendations. Deep-tier artists (ranks 500+) represent your most niche preferences and can uncover highly specialized music recommendations.

**Performance Considerations**
Lower-tier artists typically have fewer similar artists in the Last.fm database, which can result in faster processing but potentially fewer recommendations. Higher-tier artists often have extensive similarity data, leading to more recommendations but longer processing times. The system automatically adjusts API call strategies based on the selected tier to optimize performance.

**Practical Examples**

```bash
# Discover music similar to your mainstream favorites
python music.py --artist-tier-start 1 --artist-tier-end 25 --num-recs 30

# Explore recommendations based on your mid-tier preferences
python music.py --artist-tier-start 150 --artist-tier-end 250 --num-recs 25

# Find music similar to your most niche artists
python music.py --artist-tier-start 500 --artist-tier-end 700 --num-recs 20

# Quick discovery from recently played but not top-tier artists
python music.py --artist-tier-start 50 --artist-tier-end 100 --num-recs 15
```

### **Configuration Management**

The system provides multiple configuration methods to accommodate different security requirements and deployment scenarios.

**Environment Variable Configuration**
Environment variables provide the most straightforward approach for local development and are the recommended method for most users. This approach stores sensitive information in .env files that are automatically excluded from version control.

```bash
# Create environment configuration
mkdir -p config
cat > config/.env << 'EOF'
LASTFM_API_KEY=your_actual_lastfm_api_key
MUSICBRAINZ_USER_AGENT=YourMusicRecommender/1.0 (your.email@example.com)
EOF

# Use environment configuration
python music.py --config-method env --num-recs 25
```

**Encrypted Configuration**
Encrypted configuration allows you to store sensitive information in encrypted files that can be safely committed to public repositories, with decryption happening at runtime using a password.

```bash
# Create encrypted configuration
python music.py --setup-config
# Choose encrypted method and enter your password

# Use encrypted configuration
python music.py --config-method encrypted --num-recs 25
# Enter decryption password when prompted
```

**Interactive Configuration**
Interactive configuration prompts for sensitive information at runtime, providing maximum security for one-time use or highly sensitive environments.

```bash
# Use interactive prompts
python music.py --config-method prompt --num-recs 25
# Enter API key and user agent when prompted
```

### **Global Configuration Management**

Global configuration provides centralized management of system-wide settings that persist across all runs of the application.

**Data Folder Management**
The global data folder setting eliminates the need to specify data paths repeatedly and ensures consistency across different components of the system.

```bash
# Set global data folder for all future runs
python music.py --set-global-data-folder /path/to/your/spotify/data

# View current global configuration
python music.py --show-global-config

# Override global setting for a single run
python music.py --data-folder /different/path --num-recs 20
```

**System Parameters**
Global configuration also manages system-wide parameters such as cache settings, API rate limits, and default recommendation parameters.

```json
{
  "data_folder": "data/spotify",
  "cache_folder": "cache",
  "output_folder": "output",
  "logs_folder": "logs",
  "default_artist_tier_start": 1,
  "default_artist_tier_end": 50,
  "max_api_calls_per_session": 1000,
  "enable_caching": true,
  "cache_expiry_hours": 24
}
```

### **Advanced Analysis Options**

The system provides comprehensive analysis capabilities that help you understand your listening patterns and the effectiveness of different recommendation strategies.

**Listening Pattern Analysis**
Comprehensive analysis of your listening history reveals patterns in your musical consumption and helps optimize recommendation parameters.

```bash
# Basic listening analysis
python music.py --analyze-only

# Detailed analysis with tier distribution
python music.py --analyze-only --verbose

# Analysis with specific artist tier focus
python music.py --analyze-only --artist-tier-start 100 --artist-tier-end 200 --verbose
```

**Tier Distribution Analysis**
Understanding how your listening time is distributed across different artist tiers helps inform tier selection strategies for recommendation generation.

The analysis shows listening time distribution across predefined tier ranges: top 10 artists, top 11-50, top 51-100, and so on. This information reveals whether your listening is concentrated among a few favorite artists or distributed more evenly across your library, which influences the optimal tier selection strategy for recommendations.

**Temporal Pattern Analysis**
The system analyzes how your musical preferences have evolved over time, identifying periods of musical exploration, genre shifts, and preference stability. This temporal analysis informs the temporal collaborative filtering engine and helps predict future musical interests.

## ⚙️ Configuration Options

### **Command Line Parameters**

The system provides extensive command-line configuration options to customize behavior for different use cases and environments.

**Core Parameters**
- `--data-folder`: Specify Spotify data directory (overrides global configuration)
- `--config-method`: Choose configuration loading method (env, encrypted, prompt)
- `--num-recs`: Number of recommendations to generate (default: 20)
- `--analyze-only`: Perform analysis without generating recommendations
- `--verbose`: Enable detailed output and progress information

**Artist Tier Parameters**
- `--artist-tier-start`: Starting rank of artists to use as seeds (default: 1)
- `--artist-tier-end`: Ending rank of artists to use as seeds (default: 50)

**Global Configuration Parameters**
- `--set-global-data-folder`: Set global data folder and exit
- `--show-global-config`: Display current global configuration

**Setup and Testing Parameters**
- `--setup-config`: Interactive setup for secure configuration
- `--test-config`: Test configuration loading without running analysis

### **Configuration Files**

**Global Configuration (config/global_config.json)**
Stores system-wide settings that persist across all application runs. This file is automatically created with sensible defaults and can be modified directly or through command-line parameters.

**Environment Configuration (config/.env)**
Contains sensitive information such as API keys and user agents. This file should never be committed to version control and is automatically excluded by the provided .gitignore configuration.

**Encrypted Configuration (config/secrets.enc)**
Stores encrypted sensitive information that can be safely committed to public repositories. Decryption requires a password that should be shared through secure channels separate from the repository.

### **Security Configuration**

**Encryption Methods**
The system supports multiple encryption methods to accommodate different security requirements and deployment scenarios.

Fernet encryption provides symmetric encryption with password-based key derivation using PBKDF2 with SHA-256 hashing and 100,000 iterations. This method offers strong security while maintaining reasonable performance for interactive use.

AES-256-CBC encryption offers maximum security using the Advanced Encryption Standard with 256-bit keys and Cipher Block Chaining mode. This method provides the highest level of security but requires more computational resources for encryption and decryption operations.

**Key Management**
The system implements secure key management practices including secure random salt generation, proper key derivation functions, and automatic key rotation capabilities. Keys are never stored in plaintext and are derived from user passwords using industry-standard key derivation functions.

**Access Control**
Configuration loading follows a strict priority hierarchy that ensures secure defaults while providing flexibility for different deployment scenarios. Environment variables take precedence over file-based configuration, allowing for secure override in production environments.

## 🔧 Troubleshooting

### **Common Issues and Solutions**

**Data Loading Problems**
If the system cannot find your Spotify data files, verify that the files are named correctly (Audio_YYYY.json) and located in the specified data directory. Check that the global data folder configuration points to the correct location and that the files contain valid JSON data.

**API Connection Issues**
API connection problems typically result from network connectivity issues, invalid API keys, or rate limiting. Verify your internet connection and API key validity. The system includes automatic retry logic and rate limiting, but persistent issues may require checking your Last.fm API key status.

**Memory and Performance Issues**
Large Spotify datasets can consume significant memory during processing. If you encounter memory issues, consider processing data in smaller chunks or increasing available system memory. The system includes memory optimization features, but very large datasets (10+ years of heavy listening) may require additional resources.

**Configuration Loading Errors**
Configuration loading errors often result from incorrect file permissions, malformed configuration files, or missing dependencies. Verify that configuration files are readable and contain valid JSON or environment variable syntax. Check that all required dependencies are installed and up to date.

### **Debug and Diagnostic Tools**

**Verbose Output**
Enable verbose output to see detailed information about system operation, including data loading progress, API call status, and recommendation generation steps.

```bash
python music.py --verbose --analyze-only
```

**Configuration Testing**
Test configuration loading without running the full analysis to quickly identify configuration issues.

```bash
python music.py --test-config
```

**System Information**
View current system configuration and status to verify that all components are properly configured.

```bash
python music.py --show-global-config --verbose
```

### **Performance Optimization**

**Caching Strategy**
The system implements intelligent caching to reduce API calls and improve performance for repeated operations. Cache settings can be adjusted in the global configuration to balance performance and data freshness.

**API Rate Limiting**
Automatic rate limiting prevents API quota exhaustion and ensures reliable operation. The system monitors API response times and adjusts request frequency to stay within service limits while maximizing throughput.

**Memory Management**
Large datasets are processed using memory-efficient algorithms and data structures. The system includes automatic memory monitoring and garbage collection to prevent memory exhaustion during long-running operations.

## 🏗️ Architecture

### **System Components**

**Data Processing Layer**
The data processing layer handles ingestion, cleaning, and transformation of Spotify listening history data. This layer implements robust error handling for malformed data, efficient memory usage for large datasets, and comprehensive data validation to ensure analysis accuracy.

**Recommendation Engine Layer**
The recommendation engine layer contains four specialized engines that work together to generate comprehensive recommendations. Each engine focuses on different aspects of musical preference modeling and contributes unique insights to the final recommendation set.

**Security Layer**
The security layer provides comprehensive protection for sensitive information including API keys, user data, and system configuration. This layer implements multiple encryption methods, secure key management, and comprehensive access controls.

**Configuration Management Layer**
The configuration management layer handles system-wide settings, user preferences, and runtime parameters. This layer provides centralized configuration with secure defaults and flexible override capabilities.

### **Data Flow**

**Input Processing**
Raw Spotify JSON files are loaded and validated, with comprehensive error handling for common data issues. The system processes multiple years of data efficiently while maintaining memory usage within reasonable bounds.

**Feature Engineering**
Raw listening data is transformed into meaningful features that capture musical preferences, temporal patterns, and engagement metrics. This process includes sophisticated algorithms for calculating engagement scores, preference weights, and temporal trends.

**Recommendation Generation**
Multiple recommendation engines process the engineered features to generate diverse recommendation sets. Each engine contributes unique perspectives on musical preference modeling, and the results are combined using sophisticated ensemble methods.

**Output Generation**
Final recommendations are ranked, filtered, and formatted for presentation. The system provides multiple output formats and includes comprehensive metadata about recommendation sources and confidence scores.

### **Security Architecture**

**Defense in Depth**
The security architecture implements multiple layers of protection to ensure that sensitive information remains secure throughout the system lifecycle. This includes encryption at rest, secure transmission, and comprehensive access controls.

**Threat Modeling**
The system addresses common security threats including accidental secret exposure, unauthorized access to personal data, and man-in-the-middle attacks on API communications. Security measures are designed to provide protection against both accidental and malicious threats.

**Compliance and Best Practices**
The security implementation follows industry best practices for personal data protection, API key management, and secure software development. The system is designed to comply with common privacy regulations and security standards.

## 🤝 Contributing

### **Development Setup**

Contributors can set up a development environment using the same installation process as end users, with additional tools for testing and development. The system includes comprehensive testing capabilities and development utilities.

**Code Style and Standards**
The project follows Python PEP 8 style guidelines with additional conventions for security-sensitive code. All contributions should include appropriate documentation, error handling, and security considerations.

**Testing Framework**
The system includes comprehensive testing capabilities covering unit tests, integration tests, and security tests. Contributors should ensure that all new features include appropriate test coverage.

### **Feature Requests and Bug Reports**

Feature requests should include detailed use cases and implementation considerations. Bug reports should include reproduction steps, system information, and relevant log output.

**Security Issues**
Security issues should be reported through secure channels and will be addressed with high priority. The project maintains a responsible disclosure policy for security vulnerabilities.

## 📄 License

This project is released under the MIT License, which allows for both personal and commercial use with appropriate attribution. The license provides broad permissions while maintaining copyright protection and disclaimer of warranties.

## 🙏 Acknowledgments

This project builds upon the excellent work of the Spotify API team, Last.fm API developers, and the open-source Python ecosystem. Special recognition goes to the developers of the pandas, scikit-learn, and cryptography libraries that form the foundation of this system.

The recommendation algorithms are inspired by research in collaborative filtering, content-based recommendation systems, and temporal preference modeling. The security implementation follows best practices established by the cryptography and information security communities.

## 📚 References and Further Reading

For users interested in understanding the theoretical foundations of the recommendation algorithms, several academic papers and resources provide deeper insights into collaborative filtering, content-based recommendation systems, and temporal preference modeling.

The security implementation is based on established cryptographic standards and best practices for personal data protection. Users interested in the security aspects can refer to NIST guidelines for cryptographic key management and OWASP recommendations for secure application development.

The Last.fm API documentation provides comprehensive information about music metadata and similarity algorithms used by the external data sources. Understanding these APIs can help users optimize their recommendation strategies and troubleshoot API-related issues.

