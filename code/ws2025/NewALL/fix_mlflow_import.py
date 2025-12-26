"""
Helper script to fix mlflow circular import issue
This script imports mlflow in a way that avoids the circular import problem
"""
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Try to import mlflow.version first to break circular import
try:
    # Import version module directly
    import mlflow.version as mlflow_version
    print("Successfully imported mlflow.version")
except Exception as e:
    print(f"Warning: Could not import mlflow.version: {e}")

# Now try to import mlflow
try:
    import mlflow
    print("Successfully imported mlflow")
except Exception as e:
    print(f"Error importing mlflow: {e}")
    sys.exit(1)

# Try to import neuromancer components
try:
    from neuromancer.system import Node, System
    from neuromancer.problem import Problem
    from neuromancer.modules import blocks
    from neuromancer.loss import PenaltyLoss
    from neuromancer import variable
    print("Successfully imported neuromancer components")
except Exception as e:
    print(f"Error importing neuromancer: {e}")
    sys.exit(1)

print("All imports successful!")

