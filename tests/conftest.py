"""
Pytest configuration and shared fixtures for EV Charging tests.
"""
import pytest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
