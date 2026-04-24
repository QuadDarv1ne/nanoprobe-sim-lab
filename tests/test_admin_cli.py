import os
import subprocess
import sys


def test_admin_cli_help():
    """Test that admin_cli.py --help runs without error"""
    result = subprocess.run(
        [sys.executable, "admin_cli.py", "--help"], capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_admin_cli_info():
    """Test that admin_cli.py info runs without error"""
    result = subprocess.run(
        [sys.executable, "admin_cli.py", "info"], capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0
    # The output is logged to stderr due to logging configuration
    assert "Nanoprobe Sim Lab - Информация" in result.stderr


def test_admin_cli_status():
    """Test that admin_cli.py status runs without error"""
    result = subprocess.run(
        [sys.executable, "admin_cli.py", "status"], capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0
    # The output is logged to stderr due to logging configuration
    assert "Статус системы Nanoprobe Sim Lab" in result.stderr


if __name__ == "__main__":
    test_admin_cli_help()
    test_admin_cli_info()
    test_admin_cli_status()
    print("All tests passed!")
