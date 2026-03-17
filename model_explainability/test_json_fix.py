#!/usr/bin/env python3
"""
Test JSON serialization fix
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import analyze_code_metrics, map_code_metrics_to_features

def test_json_serialization():
    """Test that all returned values are JSON serializable"""
    
    test_code = """
def test_function():
    for i in range(10):
        if i % 2 == 0:
            print(i)
    return True
"""
    
    try:
        # Test code metrics
        metrics = analyze_code_metrics(test_code)
        json.dumps(metrics)  # This will raise an exception if not serializable
        print("✓ Code metrics are JSON serializable")
        
        # Test features
        features = map_code_metrics_to_features(metrics)
        json.dumps(features)  # This will raise an exception if not serializable
        print("✓ Features are JSON serializable")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    test_json_serialization()