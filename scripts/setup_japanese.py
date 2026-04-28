#!/usr/bin/env python3
"""
Setup script for Japanese language processing components.
Downloads and configures MeCab dictionaries and other Japanese NLP tools.
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import argparse
from typing import Optional

# Configure paths
BASE_DIR = Path(__file__).parent
DICT_DIR = BASE_DIR / "japanese_dicts"
MECAB_DIR = DICT_DIR / "mecab"
UNIDIC_DIR = MECAB_DIR / "unidic"


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_system_dependencies():
    """Check if required system dependencies are installed."""
    dependencies = {
        'mecab': 'MeCab morphological analyzer',
        'mecab-config': 'MeCab configuration tool',
    }

    missing = []
    for cmd, description in dependencies.items():
        result = run_command(f"which {cmd}", check=False)
        if result.returncode != 0:
            missing.append((cmd, description))

    if missing:
        print("Missing system dependencies:")
        for cmd, desc in missing:
            print(f"  - {cmd}: {desc}")
        print("\nInstallation instructions:")
        print("Ubuntu/Debian: sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8")
        print("macOS: brew install mecab mecab-ipadic")
        print("Windows: Download from https://taku910.github.io/mecab/")
        return False

    print("All system dependencies are installed.")
    return True


def download_file(url: str, destination: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {url} to {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {destination}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract archive to specified directory."""
    try:
        print(f"Extracting {archive_path} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.gz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False

        print(f"Extracted {archive_path}")
        return True
    except Exception as e:
        print(f"Failed to extract {archive_path}: {e}")
        return False


def setup_unidic():
    """Download and setup UniDic dictionary."""
    print("Setting up UniDic dictionary...")

    # UniDic Lite (smaller, faster download)
    unidic_url = "https://github.com/polm/unidic-lite/archive/refs/heads/master.zip"
    unidic_zip = DICT_DIR / "unidic-lite.zip"

    if not download_file(unidic_url, unidic_zip):
        print("Failed to download UniDic. Trying alternative method...")
        return install_unidic_pip()

    if not extract_archive(unidic_zip, UNIDIC_DIR):
        return install_unidic_pip()

    # Clean up
    unidic_zip.unlink(missing_ok=True)
    print("UniDic dictionary setup complete.")
    return True


def install_unidic_pip():
    """Install UniDic using pip as fallback."""
    print("Installing UniDic via pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "unidic-lite"], check=True)
        print("UniDic installed via pip.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install UniDic via pip: {e}")
        return False


def setup_argos_translate():
    """Setup Argos Translate for offline translation."""
    print("Setting up Argos Translate...")

    try:
        # Install Argos Translate if not already installed
        subprocess.run([sys.executable, "-m", "pip", "install", "argostranslate"], check=True)

        # Download Japanese-English translation package
        print("Downloading Japanese-English translation package...")
        import argostranslate.package
        import argostranslate.translate

        # Update package index
        argostranslate.package.update_package_index()

        # Get available packages
        available_packages = argostranslate.package.get_available_packages()

        # Find Japanese to English package
        ja_en_package = None
        for package in available_packages:
            if package.from_code == 'ja' and package.to_code == 'en':
                ja_en_package = package
                break

        if ja_en_package:
            print(f"Installing package: {ja_en_package}")
            argostranslate.package.install_from_path(ja_en_package.download())
            print("Japanese-English translation package installed.")
        else:
            print("Japanese-English translation package not found.")
            return False

        return True
    except Exception as e:
        print(f"Failed to setup Argos Translate: {e}")
        return False


def test_japanese_processing():
    """Test Japanese processing components."""
    print("\nTesting Japanese processing components...")

    # Test MeCab
    try:
        import MeCab
        tagger = MeCab.Tagger()
        result = tagger.parse("武道")
        print(f"MeCab test: {'PASS' if result else 'FAIL'}")
    except ImportError:
        print("MeCab test: FAIL (not installed)")
    except Exception as e:
        print(f"MeCab test: FAIL ({e})")

    # Test pykakasi
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        kks.setMode("H", "a")  # Hiragana to ASCII
        kks.setMode("K", "a")  # Katakana to ASCII
        kks.setMode("J", "a")  # Japanese to ASCII
        conv = kks.getConverter()
        result = conv.do("武道")
        print(f"pykakasi test: {'PASS' if result else 'FAIL'}")
    except ImportError:
        print("pykakasi test: FAIL (not installed)")
    except Exception as e:
        print(f"pykakasi test: FAIL ({e})")

    # Test Argos Translate
    try:
        import argostranslate.translate
        languages = argostranslate.translate.get_installed_languages()
        ja_installed = any(lang.code == 'ja' for lang in languages)
        en_installed = any(lang.code == 'en' for lang in languages)
        if ja_installed and en_installed:
            result = argostranslate.translate.translate("武道", "ja", "en")
            print(f"Argos Translate test: {'PASS' if result else 'FAIL'}")
        else:
            print("Argos Translate test: FAIL (languages not installed)")
    except ImportError:
        print("Argos Translate test: FAIL (not installed)")
    except Exception as e:
        print(f"Argos Translate test: FAIL ({e})")


def create_config_file():
    """Create Japanese processing configuration file."""
    config_content = '''"""
Japanese processing configuration.
Auto-generated by setup_japanese.py
"""

# MeCab configuration
MECAB_CONFIG = {
    'dicdir': None,  # Auto-detect
    'userdic': None,
    'output_format_type': 'wakati',
}

# Dictionary preferences
DICTIONARY_PREFERENCE = [
    'unidic-lite',
    'unidic',
    'ipadic',
]

# Romanization settings
ROMANIZATION = {
    'system': 'hepburn',  # hepburn, kunrei, nihon
    'use_long_vowels': True,
    'capitalize': False,
}

# Translation settings
TRANSLATION = {
    'engine': 'argos',
    'cache_translations': True,
    'fallback_to_romaji': True,
}
'''

    config_file = BASE_DIR / "japanese_config.py"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"Created configuration file: {config_file}")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Japanese language processing')
    parser.add_argument('--skip-deps', action='store_true',
                        help='Skip system dependency check')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run tests, skip setup')
    parser.add_argument('--no-argos', action='store_true',
                        help='Skip Argos Translate setup')

    args = parser.parse_args()

    if args.test_only:
        test_japanese_processing()
        return

    print("Setting up Japanese language processing for Martial Arts OCR")
    print("=" * 60)

    # Check system dependencies
    if not args.skip_deps:
        if not check_system_dependencies():
            print("\nPlease install system dependencies and run again.")
            sys.exit(1)

    # Create directories
    DICT_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    # Setup UniDic
    if not setup_unidic():
        print("WARNING: UniDic setup failed")
        success = False

    # Setup Argos Translate
    if not args.no_argos:
        if not setup_argos_translate():
            print("WARNING: Argos Translate setup failed")
            success = False

    # Create config file
    create_config_file()

    # Test everything
    test_japanese_processing()

    print("\n" + "=" * 60)
    if success:
        print("Japanese language processing setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the application")
        print("2. Upload Japanese text images to test OCR")
        print("3. Check the web interface for romanization and translation")
    else:
        print("Setup completed with some warnings.")
        print("The application may still work with reduced functionality.")

    print(f"\nConfiguration saved to: {BASE_DIR / 'japanese_config.py'}")
    print(f"Dictionaries installed to: {DICT_DIR}")


if __name__ == "__main__":
    main()