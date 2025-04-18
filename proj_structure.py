import os
import argparse
from pathlib import Path
from typing import Optional # <--- ADD THIS IMPORT

# --- Configuration: Folders/Files to Exclude ---
# Add any other specific folder or file names you want to ignore
DEFAULT_EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "mlruns",
    "artifacts",
    # "data", # Uncomment if your data dir doesn't contain relevant structure/scripts
    ".vscode",
    ".idea", # Common IDE folder
    ".pytest_cache",
    "node_modules", # Common web dev folder
    "build",
    "dist",
    ".egg-info",
    "htmlcov", # Coverage reports
    ".mypy_cache",
    ".ruff_cache",
    # Add specific files/dirs from your project if needed
    "logs", # Example: Exclude Airflow logs if they are in the root
}

DEFAULT_EXCLUDE_FILES = {
    ".DS_Store", # macOS specific
    "Thumbs.db", # Windows specific
    ".env", # Common environment file
    "docker-compose.yaml", # Often less relevant for code structure view
    "docker-compose.yml",
    "Dockerfile",
    "setup.cfg",
    "pyproject.toml", # Config files, sometimes hide them for pure structure
    "requirements.txt",
    "constraints.txt", # Airflow constraints file
}
# --- End Configuration ---

def print_tree(
    directory: Path,
    prefix: str = "",
    exclude_dirs: set = DEFAULT_EXCLUDE_DIRS,
    exclude_files: set = DEFAULT_EXCLUDE_FILES,
    level: int = -1, # -1 indicates root level, adjust print accordingly
    limit_depth: Optional[int] = None, # <--- CHANGE THIS LINE
    print_files: bool = True
):
    """
    Recursively prints a directory tree structure.

    Args:
        directory (Path): The directory path to start from.
        prefix (str): The prefix string for indentation and tree lines.
        exclude_dirs (set): A set of directory names to exclude.
        exclude_files (set): A set of file names to exclude.
        level (int): Current recursion depth.
        limit_depth (Optional[int]): Maximum depth to traverse. None means no limit. # <-- Docstring update
        print_files (bool): Whether to include files in the output.
    """
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    if level == -1: # Print root directory name cleanly
        print(f"{directory.name}/")
        level = 0 # Start counting depth from here

    if limit_depth is not None and level >= limit_depth:
        return

    try:
        # Get directory contents, handling potential permission errors
        contents = list(directory.iterdir())
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return
    except FileNotFoundError:
         print(f"{prefix}└── [Not Found - Possibly a broken symlink]")
         return


    # Separate and filter directories and files
    dirs = sorted([
        item for item in contents
        if item.is_dir() and item.name not in exclude_dirs
    ])
    files = sorted([
        item for item in contents
        if item.is_file() and item.name not in exclude_files
    ])

    # Combine filtered entries (directories first, then files if printing them)
    entries = dirs + (files if print_files else [])

    pointers = ["├── "] * (len(entries) - 1) + ["└── "]

    for pointer, entry in zip(pointers, entries):
        # Skip printing the script itself if it's in the directory
        if entry.name == Path(__file__).name:
             continue

        if entry.is_dir():
            yield prefix + pointer + entry.name + "/"
            # Decide the extension for the recursive call's prefix
            extension = "│   " if pointer == "├── " else "    "
            yield from print_tree(
                entry,
                prefix + extension,
                exclude_dirs,
                exclude_files,
                level + 1,
                limit_depth,
                print_files
            )
        elif print_files:
            yield prefix + pointer + entry.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a directory tree structure, excluding common non-code folders.",
        formatter_class=argparse.RawTextHelpFormatter # Keep newline in description
    )
    parser.add_argument(
        "start_path",
        nargs="?",
        default=".",
        help="The starting directory path (default: current directory).",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_DIRS), # Use default list if not provided
        help=f"Additional directory names to exclude. Defaults: {', '.join(DEFAULT_EXCLUDE_DIRS)}",
    )
    parser.add_argument(
        "--exclude-files",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_FILES), # Use default list if not provided
        help=f"Additional file names to exclude. Defaults: {', '.join(DEFAULT_EXCLUDE_FILES)}",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Limit the depth of the directory traversal.",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="Only show directories, do not list files.",
    )

    args = parser.parse_args()

    start_directory = Path(args.start_path).resolve() # Use absolute path
    exclude_dirs_set = set(args.exclude_dirs)
    exclude_files_set = set(args.exclude_files)

    if not start_directory.is_dir():
        print(f"Error: Starting path '{args.start_path}' is not a valid directory.")
    else:
        # Use a generator and print line by line
        # Also add the script's own name to the exclude list dynamically
        script_filename = Path(__file__).name
        exclude_files_set.add(script_filename)

        for line in print_tree(
            directory=start_directory,
            exclude_dirs=exclude_dirs_set,
            exclude_files=exclude_files_set,
            limit_depth=args.depth,
            print_files=not args.no_files
        ):
            print(line)