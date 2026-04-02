"""
nova.fs — File system operations constellation
Core I/O, path manipulation, directory traversal
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

__all__ = [
    'read_file', 'write_file', 'append_file',
    'file_exists', 'delete_file', 'list_dir', 'mkdir', 'mkdir_recursive',
    'is_file', 'is_dir', 'get_size', 'get_modified_time',
    'current_dir', 'change_dir', 'copy_file', 'move_file'
]


def read_file(path: str) -> str:
    """
    Read entire file into string.
    
    Args:
        path: File path
    
    Returns:
        File contents as string
    
    Raises:
        FileNotFoundError: If file does not exist
        IOError: On read error
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise IOError(f"Error reading {path}: {e}")


def write_file(path: str, content: str) -> None:
    """
    Write string to file (overwrites if exists).
    
    Args:
        path: File path
        content: Content to write
    
    Raises:
        IOError: On write error
    """
    try:
        # Create parent directory if needed
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Error writing to {path}: {e}")


def append_file(path: str, content: str) -> None:
    """
    Append string to file.
    
    Args:
        path: File path
        content: Content to append
    
    Raises:
        IOError: On write error
    """
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Error appending to {path}: {e}")


def file_exists(path: str) -> bool:
    """
    Check if file exists.
    
    Args:
        path: File path
    
    Returns:
        True if file exists, False otherwise
    """
    return os.path.isfile(path)


def delete_file(path: str) -> Tuple[bool, Optional[str]]:
    """
    Delete file.
    
    Args:
        path: File path
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    try:
        os.remove(path)
        return (True, None)
    except FileNotFoundError:
        return (False, f"File not found: {path}")
    except Exception as e:
        return (False, f"Error deleting {path}: {e}")


def list_dir(path: str) -> List[str]:
    """
    List directory contents.
    
    Args:
        path: Directory path
    
    Returns:
        List of filenames (Array[Str] in NOVA)
    
    Raises:
        FileNotFoundError: If directory does not exist
        NotADirectoryError: If path is not a directory
    """
    try:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Not a directory: {path}")
        return os.listdir(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {path}")
    except Exception as e:
        raise IOError(f"Error listing {path}: {e}")


def mkdir(path: str) -> Tuple[bool, Optional[str]]:
    """
    Create single directory (parent must exist).
    
    Args:
        path: Directory path
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    try:
        os.mkdir(path)
        return (True, None)
    except FileExistsError:
        return (False, f"Directory already exists: {path}")
    except FileNotFoundError:
        return (False, f"Parent directory does not exist")
    except Exception as e:
        return (False, f"Error creating {path}: {e}")


def mkdir_recursive(path: str) -> Tuple[bool, Optional[str]]:
    """
    Create directory and all parent directories.
    
    Args:
        path: Directory path
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    try:
        os.makedirs(path, exist_ok=True)
        return (True, None)
    except Exception as e:
        return (False, f"Error creating {path}: {e}")


def is_file(path: str) -> bool:
    """
    Check if path is a regular file.
    
    Args:
        path: Path
    
    Returns:
        True if path is a file
    """
    return os.path.isfile(path)


def is_dir(path: str) -> bool:
    """
    Check if path is a directory.
    
    Args:
        path: Path
    
    Returns:
        True if path is a directory
    """
    return os.path.isdir(path)


def get_size(path: str) -> int:
    """
    Get file/directory size in bytes.
    
    Args:
        path: File or directory path
    
    Returns:
        Size in bytes
    
    Raises:
        FileNotFoundError: If path does not exist
    """
    try:
        return os.path.getsize(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Path not found: {path}")
    except Exception as e:
        raise IOError(f"Error getting size of {path}: {e}")


def get_modified_time(path: str) -> float:
    """
    Get file/directory modification time.
    
    Args:
        path: Path
    
    Returns:
        Unix timestamp (Float[1] in NOVA, seconds since epoch)
    
    Raises:
        FileNotFoundError: If path does not exist
    """
    try:
        return float(os.path.getmtime(path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Path not found: {path}")
    except Exception as e:
        raise IOError(f"Error getting mtime of {path}: {e}")


def current_dir() -> str:
    """
    Get current working directory.
    
    Returns:
        Current directory path
    """
    return os.getcwd()


def change_dir(path: str) -> Tuple[bool, Optional[str]]:
    """
    Change current working directory.
    
    Args:
        path: Directory path to change to
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    try:
        os.chdir(path)
        return (True, None)
    except FileNotFoundError:
        return (False, f"Directory not found: {path}")
    except Exception as e:
        return (False, f"Error changing to {path}: {e}")


def copy_file(src: str, dst: str) -> Tuple[bool, Optional[str]]:
    """
    Copy file from src to dst.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    try:
        shutil.copy2(src, dst)
        return (True, None)
    except FileNotFoundError:
        return (False, f"Source file not found: {src}")
    except Exception as e:
        return (False, f"Error copying {src} to {dst}: {e}")


def move_file(src: str, dst: str) -> Tuple[bool, Optional[str]]:
    """
    Move or rename file from src to dst.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    try:
        shutil.move(src, dst)
        return (True, None)
    except FileNotFoundError:
        return (False, f"Source file not found: {src}")
    except Exception as e:
        return (False, f"Error moving {src} to {dst}: {e}")
