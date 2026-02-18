"""
This file centralizes project path management.

It automatically detects the repository root by looking for the `.git` folder 
and defines key directories relative to the project root, including:

- PROJECT_ROOT: the root directory of the repository

This setup ensures that all scripts and notebooks can use consistent paths 
without hardcoding absolute directories, making the project portable across 
different machines.

Subdirectories such as data, models, or outputs can be defined relative to PROJECT_ROOT, e.g.:

    from src.config import PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"

This setup ensures that all scripts and notebooks can use consistent paths 
without hardcoding absolute directories, making the project portable across 
different machines.
"""

from pathlib import Path

# Auto-detect repo root 
def find_repo_root():
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd() 

# Define root path to allow importing
PROJECT_ROOT = find_repo_root()