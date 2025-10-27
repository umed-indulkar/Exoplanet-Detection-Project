"""
Simple test - checks if basic imports work
Run this FIRST before the full test
"""

print("Testing imports...")

try:
    print("1. Testing numpy...")
    import numpy as np
    print("   ✓ numpy OK")
except Exception as e:
    print(f"   ✗ numpy FAILED: {e}")
    print("   Fix: pip install numpy")

try:
    print("2. Testing pandas...")
    import pandas as pd
    print("   ✓ pandas OK")
except Exception as e:
    print(f"   ✗ pandas FAILED: {e}")
    print("   Fix: pip install pandas")

try:
    print("3. Testing scipy...")
    from scipy import stats
    print("   ✓ scipy OK")
except Exception as e:
    print(f"   ✗ scipy FAILED: {e}")
    print("   Fix: pip install scipy")

try:
    print("4. Testing exodet core...")
    from exodet.core import exceptions
    print("   ✓ exodet.core OK")
except Exception as e:
    print(f"   ✗ exodet.core FAILED: {e}")
    print(f"   Error: {e}")

try:
    print("5. Testing exodet imports...")
    from exodet import Config
    print("   ✓ Config OK")
    
    from exodet.core.data_loader import LightCurve
    print("   ✓ LightCurve OK")
    
    print("\n✅ ALL IMPORTS WORKING!")
    print("\nYou can now run: python test_unified_system.py")
    
except Exception as e:
    print(f"   ✗ exodet imports FAILED")
    print(f"   Error: {e}")
    print(f"\n   This is normal if dependencies aren't installed yet.")
    print(f"   Run: pip install -r requirements.txt")

print("\nDone!")
