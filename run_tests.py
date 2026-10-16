"""
Direct test runner script to execute tests without any shell escaping issues.
"""
import sys
import os

print(f"Executing tests using Python: {sys.executable}")
print(f"Python version: {sys.version}")

# Ensure workspace root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest

# Run pytest directly
exit_code = pytest.main(["-v", "tests", "-s", "-p", "no:anyio"])
print(f"\nPytest exited with status code: {exit_code}")
sys.exit(exit_code)
