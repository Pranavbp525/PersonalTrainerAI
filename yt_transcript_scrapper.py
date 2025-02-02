import os
from pathlib import Path

DATA_FOLDER = 'data'  # Folder to store JSON files

def setup_data_folder():
    """Create data folder if it doesn't exist."""
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function."""
    setup_data_folder()

if __name__ == "__main__":
    main()
