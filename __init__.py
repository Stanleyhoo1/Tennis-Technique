import sys
from pathlib import Path
import os

# Add paths to sys.path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
sys.path.insert(1, str(root_path / "yolov7"))

print("Current Working Directory:", os.getcwd())

# Import from setup after paths are set
from setup import *
model = get_model()

# Import other modules after paths are configured
from classification import classify_swing
from overlay import overlay_swing_path

# Clean up sys.path to remove added paths
sys.path.pop(0)
sys.path.pop(0)
