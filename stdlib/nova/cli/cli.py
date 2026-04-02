"""
nova.cli — Command-line interface constellation
Argument parsing, help generation, interactive input
"""

import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple

__all__ = [
    'add_argument', 'parse_args', 'get_arg', 'get_arg_int', 'get_arg_float', 'get_arg_bool',
    'get_arg_list', 'has_arg', 'set_help_text', 'print_help', 'print_usage',
    'set_version', 'print_version', 'input_required', 'input_optional',
    'confirm', 'progress_bar'
]

# Global argument parser state
_parser = None
_args = None
_program_name = "nova"
_version = "0.1.0"
_help_text = ""


def _get_parser():
    """Ensure parser exists."""
    global _parser
    if _parser is None:
        _parser = argparse.ArgumentParser(description=_help_text, add_help=True)
    return _parser


def add_argument(name: str, type_name: str = "str",
                short: str = "", long: str = "",
                required: bool = False, help_text: str = "") -> None:
    """
    Add an argument to the parser.
    
    Args:
        name: Argument name (used for storage)
        type_name: 'str', 'int', 'float', 'bool', 'list'
        short: Short flag (e.g., '-v')
        long: Long flag (e.g., '--verbose')
        required: Whether argument is required
        help_text: Help text for this argument
    """
    parser = _get_parser()
    
    # Determine type converter
    type_converter = str
    if type_name == "int":
        type_converter = int
    elif type_name == "float":
        type_converter = float
    elif type_name == "bool":
        type_converter = lambda x: x.lower() in ('true', '1', 'yes')
    
    # Build flag list
    flags = []
    if short:
        flags.append(short)
    if long:
        flags.append(long)
    if not flags:
        flags.append(f"--{name}")
    
    # Add argument
    parser.add_argument(*flags, dest=name, type=type_converter,
                       required=required, help=help_text)


def parse_args(argv: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Parse command-line arguments.
    
    Args:
        argv: Argument list (typically sys.argv[1:])
    
    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    global _args
    parser = _get_parser()
    
    try:
        _args = parser.parse_args(argv)
        return (True, None)
    except SystemExit:
        return (False, "Argument parsing failed")
    except Exception as e:
        return (False, str(e))


def get_arg(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get a string argument value.
    
    Args:
        name: Argument name
    
    Returns:
        (value, None) if found
        (None, error) if not found
    """
    if _args is None:
        return (None, "Arguments not parsed")
    
    try:
        value = getattr(_args, name, None)
        if value is None:
            return (None, f"Argument not found: {name}")
        return (str(value), None)
    except Exception as e:
        return (None, str(e))


def get_arg_int(name: str) -> Tuple[Optional[int], Optional[str]]:
    """Get an integer argument value."""
    value, error = get_arg(name)
    if error:
        return (None, error)
    try:
        return (int(value), None)
    except ValueError:
        return (None, f"Not an integer: {value}")


def get_arg_float(name: str) -> Tuple[Optional[float], Optional[str]]:
    """Get a float argument value."""
    value, error = get_arg(name)
    if error:
        return (None, error)
    try:
        return (float(value), None)
    except ValueError:
        return (None, f"Not a float: {value}")


def get_arg_bool(name: str) -> Tuple[Optional[bool], Optional[str]]:
    """Get a boolean argument value."""
    value, error = get_arg(name)
    if error:
        return (None, error)
    
    bool_value = value.lower() in ('true', '1', 'yes', 'y')
    return (bool_value, None)


def get_arg_list(name: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Get a list argument value."""
    if _args is None:
        return (None, "Arguments not parsed")
    
    try:
        value = getattr(_args, name, None)
        if value is None:
            return ([], None)  # Empty list if not provided
        if isinstance(value, list):
            return (value, None)
        return ([str(value)], None)
    except Exception as e:
        return (None, str(e))


def has_arg(name: str) -> bool:
    """
    Check if argument was provided.
    
    Args:
        name: Argument name
    
    Returns:
        True if argument was provided
    """
    if _args is None:
        return False
    
    return hasattr(_args, name) and getattr(_args, name) is not None


def set_help_text(program_name: str, help_text: str) -> None:
    """
    Set program name and help text.
    
    Args:
        program_name: Program name for help display
        help_text: Description text
    """
    global _program_name, _help_text
    _program_name = program_name
    _help_text = help_text
    
    if _parser is not None:
        _parser.description = help_text
        _parser.prog = program_name


def print_help() -> None:
    """Print help message."""
    parser = _get_parser()
    parser.print_help()


def print_usage() -> None:
    """Print usage message."""
    parser = _get_parser()
    parser.print_usage()


def set_version(version: str) -> None:
    """Set program version."""
    global _version
    _version = version


def print_version() -> None:
    """Print version."""
    print(f"{_program_name} {_version}")


def input_required(prompt: str) -> str:
    """
    Prompt for required input (loop until non-empty).
    
    Args:
        prompt: Prompt text
    
    Returns:
        User input
    """
    while True:
        response = input(prompt)
        if response.strip():
            return response
        print("Input required. Please try again.")


def input_optional(prompt: str, default: str = "") -> str:
    """
    Prompt for optional input with default.
    
    Args:
        prompt: Prompt text
        default: Default value if empty input
    
    Returns:
        User input or default
    """
    response = input(f"{prompt} [{default}]: ")
    return response if response.strip() else default


def confirm(prompt: str) -> bool:
    """
    Prompt for yes/no confirmation.
    
    Args:
        prompt: Prompt text
    
    Returns:
        True if user entered yes (y/yes/1/true)
    """
    response = input(f"{prompt} [y/N]: ")
    return response.lower() in ('y', 'yes', '1', 'true')


def progress_bar(total: int) -> callable:
    """
    Create a progress bar iterator.
    
    Args:
        total: Total number of items
    
    Returns:
        Function that takes current count and updates bar
    """
    def update(current: int) -> None:
        """Update progress bar display."""
        percent = 100 * current / total
        filled = int(50 * current / total)
        bar = '█' * filled + '░' * (50 - filled)
        print(f'\r|{bar}| {percent:.1f}%', end='', flush=True)
    
    return update
