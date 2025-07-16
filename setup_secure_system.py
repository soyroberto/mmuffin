#!/usr/bin/env python3
"""
Automated Setup Script for Secure Music Recommendation System
Handles complete setup including dependencies, configuration, and security
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_directory_structure():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = [
        "config",
        "data/spotify",
        "cache",
        "logs",
        "output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}/")
    
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    # Check if pip is available
    if not run_command("pip --version", "Checking pip availability"):
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install security requirements
    if not run_command("pip install cryptography>=41.0.0 python-dotenv>=1.0.0", 
                      "Installing security dependencies"):
        return False
    
    # Install core requirements
    core_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "implicit>=0.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    return True

def setup_gitignore():
    """Setup .gitignore file"""
    print("🔒 Setting up .gitignore for security...")
    
    gitignore_content = """# Music Recommendation System - Security and Privacy
# DO NOT COMMIT THESE FILES TO VERSION CONTROL

# Environment files containing secrets
.env
config/.env
.env.local
.env.production

# Encryption keys and salts
config/key.key
config/salt.key
*.key

# Personal data and cache
data/spotify/
cache/
logs/
output/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
music_rec_env/
music_recommender_env/

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Backup files
*.bak
*.backup
*.orig

# Temporary files
*.tmp
*.temp
temp/
tmp/
"""
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        # Append to existing .gitignore
        with open(gitignore_path, 'a') as f:
            f.write("\n" + gitignore_content)
        print("   ✅ Added security entries to existing .gitignore")
    else:
        # Create new .gitignore
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("   ✅ Created new .gitignore file")
    
    return True

def create_readme():
    """Create README with setup instructions"""
    print("📝 Creating README.md...")
    
    readme_content = """# Secure Music Recommendation System

A personal music recommendation system that analyzes your Spotify listening history and provides personalized recommendations using machine learning and external music APIs.

## 🔐 Security Features

- Encrypted API key storage
- Multiple configuration methods
- Git-safe secret management
- No hardcoded credentials

## 🚀 Quick Start

### 1. Setup Configuration
```bash
python secure_music_recommender.py --setup-config
```

### 2. Add Your Spotify Data
Place your Spotify data export JSON files in `data/spotify/`:
- `Audio_2012.json`
- `Audio_2013.json`
- etc.

### 3. Generate Recommendations
```bash
python secure_music_recommender.py --config-method env --num-recs 30
```

## 📊 Usage Examples

### Analyze Your Listening Patterns
```bash
python secure_music_recommender.py --analyze-only --verbose
```

### Get Recommendations with Different Methods
```bash
# Using environment variables
python secure_music_recommender.py --config-method env --num-recs 50

# Using encrypted configuration
python secure_music_recommender.py --config-method encrypted --num-recs 50

# Interactive prompts
python secure_music_recommender.py --config-method prompt --num-recs 50
```

## 🔧 Configuration Methods

1. **Environment Variables** (Recommended for development)
2. **Encrypted Files** (Good for shared environments)
3. **Interactive Prompts** (Secure for one-time use)

## 📁 Project Structure

```
music-recommender/
├── config/
│   ├── .env                 # Your secrets (not committed)
│   ├── env.template         # Template (safe to commit)
│   └── secrets.enc          # Encrypted secrets (optional)
├── data/spotify/            # Your Spotify JSON files
├── secure_music_recommender.py
├── secrets_encryption_system.py
└── README.md
```

## 🛡️ Security

- All sensitive data is encrypted or stored in environment variables
- No API keys are hardcoded in the source code
- Comprehensive .gitignore prevents accidental secret commits
- Multiple encryption methods available

## 📚 Documentation

See `security_setup_guide.md` for detailed setup instructions and security best practices.

## 🤝 Contributing

This is a personal project, but feel free to fork and adapt for your own use!

## 📄 License

MIT License - See LICENSE file for details
"""
    
    with open("README.md", 'w') as f:
        f.write(readme_content)
    
    print("   ✅ Created README.md")
    return True

def verify_setup():
    """Verify the setup is working correctly"""
    print("🧪 Verifying setup...")
    
    # Check if required files exist
    required_files = [
        "secrets_encryption_system.py",
        "secure_music_recommender.py",
        "fixed_recommendation_prototype.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all system files are in the current directory")
        return False
    
    # Test imports
    try:
        import cryptography
        print("   ✅ Cryptography library available")
    except ImportError:
        print("   ❌ Cryptography library not available")
        return False
    
    try:
        import pandas
        import numpy
        import sklearn
        print("   ✅ Core ML libraries available")
    except ImportError:
        print("   ❌ Some core ML libraries missing")
        return False
    
    print("   ✅ Setup verification complete")
    return True

def main():
    """Main setup function"""
    print("🎵 Secure Music Recommendation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directory structure
    if not setup_directory_structure():
        print("❌ Failed to setup directories")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing Dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("Please install dependencies manually:")
        print("pip install cryptography python-dotenv pandas numpy scikit-learn requests implicit")
        sys.exit(1)
    
    # Setup security
    if not setup_gitignore():
        print("❌ Failed to setup .gitignore")
        sys.exit(1)
    
    # Create documentation
    if not create_readme():
        print("❌ Failed to create README")
        sys.exit(1)
    
    # Verify setup
    if not verify_setup():
        print("❌ Setup verification failed")
        sys.exit(1)
    
    print("\n🎉 Setup Complete!")
    print("=" * 20)
    print("\nNext steps:")
    print("1. Add your Spotify data files to data/spotify/")
    print("2. Run: python secure_music_recommender.py --setup-config")
    print("3. Generate recommendations: python secure_music_recommender.py --config-method env --num-recs 30")
    print("\nFor detailed instructions, see security_setup_guide.md")
    print("\n🔒 Your API keys will be securely encrypted and never committed to Git!")

if __name__ == "__main__":
    main()

