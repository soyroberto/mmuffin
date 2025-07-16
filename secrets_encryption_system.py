#!/usr/bin/env python3
"""
Comprehensive Secrets Encryption System for Music Recommendation System
Provides multiple secure methods for handling API keys and sensitive configuration

Features:
- Environment variable management
- AES encryption with password-based key derivation
- Fernet encryption (symmetric)
- Secure key generation and storage
- Git-safe configuration management
"""

import os
import json
import base64
import getpass
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

class SecureConfigManager:
    """
    Comprehensive configuration manager with multiple encryption options
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # File paths
        self.encrypted_config_file = self.config_dir / "secrets.enc"
        self.key_file = self.config_dir / "key.key"
        self.salt_file = self.config_dir / "salt.key"
        self.env_template_file = self.config_dir / "env.template"
        self.env_file = self.config_dir / ".env"
        
    def generate_key_from_password(self, password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """
        Generate encryption key from password using PBKDF2
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return key, salt
    
    def generate_fernet_key(self) -> bytes:
        """
        Generate a Fernet encryption key
        """
        return Fernet.generate_key()
    
    def encrypt_with_fernet(self, data: str, key: bytes) -> bytes:
        """
        Encrypt data using Fernet (symmetric encryption)
        """
        f = Fernet(key)
        return f.encrypt(data.encode())
    
    def decrypt_with_fernet(self, encrypted_data: bytes, key: bytes) -> str:
        """
        Decrypt data using Fernet
        """
        f = Fernet(key)
        return f.decrypt(encrypted_data).decode()
    
    def encrypt_with_aes(self, data: str, key: bytes) -> tuple[bytes, bytes]:
        """
        Encrypt data using AES-256-CBC
        """
        # Generate random IV
        iv = os.urandom(16)
        
        # Pad data to multiple of 16 bytes
        pad_length = 16 - (len(data) % 16)
        padded_data = data + (chr(pad_length) * pad_length)
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data.encode()) + encryptor.finalize()
        
        return encrypted_data, iv
    
    def decrypt_with_aes(self, encrypted_data: bytes, key: bytes, iv: bytes) -> str:
        """
        Decrypt data using AES-256-CBC
        """
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        pad_length = padded_data[-1]
        data = padded_data[:-pad_length].decode()
        
        return data
    
    def create_encrypted_config(self, secrets: Dict[str, str], password: str, method: str = "fernet"):
        """
        Create encrypted configuration file
        
        Args:
            secrets: Dictionary of secret key-value pairs
            password: Password for encryption
            method: Encryption method ('fernet' or 'aes')
        """
        secrets_json = json.dumps(secrets, indent=2)
        
        if method == "fernet":
            # Generate key from password
            key, salt = self.generate_key_from_password(password)
            fernet_key = base64.urlsafe_b64encode(key)
            
            # Encrypt data
            encrypted_data = self.encrypt_with_fernet(secrets_json, fernet_key)
            
            # Save encrypted data and salt
            config_data = {
                "method": "fernet",
                "data": base64.b64encode(encrypted_data).decode(),
                "salt": base64.b64encode(salt).decode()
            }
            
        elif method == "aes":
            # Generate key from password
            key, salt = self.generate_key_from_password(password)
            
            # Encrypt data
            encrypted_data, iv = self.encrypt_with_aes(secrets_json, key)
            
            # Save encrypted data, salt, and IV
            config_data = {
                "method": "aes",
                "data": base64.b64encode(encrypted_data).decode(),
                "salt": base64.b64encode(salt).decode(),
                "iv": base64.b64encode(iv).decode()
            }
        
        else:
            raise ValueError(f"Unsupported encryption method: {method}")
        
        # Write encrypted config
        with open(self.encrypted_config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Encrypted configuration saved to {self.encrypted_config_file}")
        print(f"🔐 Encryption method: {method.upper()}")
    
    def load_encrypted_config(self, password: str) -> Dict[str, str]:
        """
        Load and decrypt configuration file
        """
        if not self.encrypted_config_file.exists():
            raise FileNotFoundError(f"Encrypted config file not found: {self.encrypted_config_file}")
        
        # Load encrypted config
        with open(self.encrypted_config_file, 'r') as f:
            config_data = json.load(f)
        
        method = config_data["method"]
        encrypted_data = base64.b64decode(config_data["data"])
        salt = base64.b64decode(config_data["salt"])
        
        # Generate key from password
        key, _ = self.generate_key_from_password(password, salt)
        
        if method == "fernet":
            fernet_key = base64.urlsafe_b64encode(key)
            decrypted_json = self.decrypt_with_fernet(encrypted_data, fernet_key)
            
        elif method == "aes":
            iv = base64.b64decode(config_data["iv"])
            decrypted_json = self.decrypt_with_aes(encrypted_data, key, iv)
            
        else:
            raise ValueError(f"Unsupported encryption method: {method}")
        
        return json.loads(decrypted_json)
    
    def create_env_file(self, secrets: Dict[str, str]):
        """
        Create .env file for environment variables (local development)
        """
        env_content = "# Environment variables for Music Recommendation System\n"
        env_content += "# DO NOT COMMIT THIS FILE TO VERSION CONTROL\n\n"
        
        for key, value in secrets.items():
            env_content += f"{key}={value}\n"
        
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Environment file created: {self.env_file}")
    
    def create_env_template(self, secrets: Dict[str, str]):
        """
        Create .env template file (safe for version control)
        """
        template_content = "# Environment variables template for Music Recommendation System\n"
        template_content += "# Copy this file to .env and fill in your actual values\n"
        template_content += "# The .env file should NOT be committed to version control\n\n"
        
        for key in secrets.keys():
            template_content += f"{key}=your_{key.lower()}_here\n"
        
        with open(self.env_template_file, 'w') as f:
            f.write(template_content)
        
        print(f"✅ Environment template created: {self.env_template_file}")
    
    def load_from_env(self) -> Dict[str, str]:
        """
        Load secrets from environment variables
        """
        secrets = {}
        
        # Try to load from .env file first
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        secrets[key] = value
        
        # Override with actual environment variables
        env_keys = ['LASTFM_API_KEY', 'MUSICBRAINZ_USER_AGENT']
        for key in env_keys:
            if key in os.environ:
                secrets[key] = os.environ[key]
        
        return secrets
    
    def create_gitignore_entries(self) -> str:
        """
        Generate .gitignore entries for security
        """
        gitignore_content = """
# Music Recommendation System - Security Files
# DO NOT COMMIT THESE FILES TO VERSION CONTROL

# Environment files
.env
config/.env

# Encryption keys
config/key.key
config/salt.key

# Encrypted secrets (optional - you may want to commit this)
# config/secrets.enc

# Cache and temporary files
cache/
*.pyc
__pycache__/
.DS_Store

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
Thumbs.db
"""
        return gitignore_content.strip()

class SecureConfigLoader:
    """
    Secure configuration loader for the music recommendation system
    """
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config_manager = config_manager
        self.secrets = {}
    
    def load_secrets(self, method: str = "env") -> Dict[str, str]:
        """
        Load secrets using specified method
        
        Args:
            method: Loading method ('env', 'encrypted', 'prompt')
        """
        if method == "env":
            self.secrets = self.config_manager.load_from_env()
            
        elif method == "encrypted":
            password = getpass.getpass("Enter decryption password: ")
            try:
                self.secrets = self.config_manager.load_encrypted_config(password)
            except Exception as e:
                print(f"❌ Failed to decrypt config: {e}")
                return {}
                
        elif method == "prompt":
            self.secrets = self._prompt_for_secrets()
            
        else:
            raise ValueError(f"Unsupported loading method: {method}")
        
        # Validate required secrets
        required_keys = ['LASTFM_API_KEY', 'MUSICBRAINZ_USER_AGENT']
        missing_keys = [key for key in required_keys if not self.secrets.get(key)]
        
        if missing_keys:
            print(f"⚠️  Missing required secrets: {missing_keys}")
            return {}
        
        print(f"✅ Successfully loaded {len(self.secrets)} secrets")
        return self.secrets
    
    def _prompt_for_secrets(self) -> Dict[str, str]:
        """
        Prompt user for secrets interactively
        """
        secrets = {}
        
        print("Enter your API credentials:")
        secrets['LASTFM_API_KEY'] = getpass.getpass("Last.fm API Key: ")
        secrets['MUSICBRAINZ_USER_AGENT'] = input("MusicBrainz User Agent: ")
        
        return secrets
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a specific secret value
        """
        return self.secrets.get(key, default)

def setup_secure_config():
    """
    Interactive setup for secure configuration
    """
    print("🔐 Music Recommendation System - Secure Configuration Setup")
    print("=" * 60)
    
    config_manager = SecureConfigManager()
    
    # Get secrets from user
    print("\n1. Enter your API credentials:")
    secrets = {
        'LASTFM_API_KEY': getpass.getpass("Last.fm API Key: "),
        'MUSICBRAINZ_USER_AGENT': input("MusicBrainz User Agent: ")
    }
    
    # Choose encryption method
    print("\n2. Choose security method:")
    print("   a) Environment variables (.env file) - Recommended for development")
    print("   b) Encrypted file (Fernet) - Good for shared environments")
    print("   c) Encrypted file (AES) - Maximum security")
    print("   d) All methods - Create all options")
    
    choice = input("Enter choice (a/b/c/d): ").lower()
    
    if choice in ['a', 'd']:
        # Create environment files
        config_manager.create_env_file(secrets)
        config_manager.create_env_template()
    
    if choice in ['b', 'd']:
        # Create Fernet encrypted file
        password = getpass.getpass("Enter encryption password: ")
        config_manager.create_encrypted_config(secrets, password, "fernet")
    
    if choice in ['c', 'd']:
        # Create AES encrypted file
        password = getpass.getpass("Enter encryption password: ")
        config_manager.create_encrypted_config(secrets, password, "aes")
    
    # Create .gitignore entries
    gitignore_content = config_manager.create_gitignore_entries()
    
    print("\n3. Add these entries to your .gitignore file:")
    print("-" * 40)
    print(gitignore_content)
    print("-" * 40)
    
    # Save .gitignore
    gitignore_file = Path(".gitignore")
    if gitignore_file.exists():
        with open(gitignore_file, 'a') as f:
            f.write("\n" + gitignore_content + "\n")
        print("✅ Added security entries to existing .gitignore")
    else:
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content + "\n")
        print("✅ Created new .gitignore file")
    
    print("\n🎉 Secure configuration setup complete!")
    print("\nNext steps:")
    print("1. Test loading your configuration")
    print("2. Commit your code (secrets are now protected)")
    print("3. Share the repository safely")

def test_configuration():
    """
    Test configuration loading
    """
    print("🧪 Testing Configuration Loading")
    print("=" * 35)
    
    config_manager = SecureConfigManager()
    loader = SecureConfigLoader(config_manager)
    
    # Test environment variables
    print("\n1. Testing environment variable loading...")
    env_secrets = loader.load_secrets("env")
    if env_secrets:
        print("✅ Environment variables loaded successfully")
        for key in env_secrets:
            print(f"   {key}: {'*' * len(env_secrets[key])}")
    else:
        print("❌ No environment variables found")
    
    # Test encrypted file (if exists)
    if config_manager.encrypted_config_file.exists():
        print("\n2. Testing encrypted file loading...")
        try:
            encrypted_secrets = loader.load_secrets("encrypted")
            if encrypted_secrets:
                print("✅ Encrypted configuration loaded successfully")
                for key in encrypted_secrets:
                    print(f"   {key}: {'*' * len(encrypted_secrets[key])}")
        except Exception as e:
            print(f"❌ Failed to load encrypted config: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            setup_secure_config()
        elif sys.argv[1] == "test":
            test_configuration()
        else:
            print("Usage: python secrets_encryption_system.py [setup|test]")
    else:
        print("🔐 Secrets Encryption System for Music Recommendation")
        print("Usage:")
        print("  python secrets_encryption_system.py setup  - Setup secure configuration")
        print("  python secrets_encryption_system.py test   - Test configuration loading")

