import sys
from pathlib import Path
import os

root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
sys.path.insert(1, str(root_path / "yolov7"))

from setup import *
model = get_model()

from classification import classify_swing
from overlay import overlay_swing_path

sys.path.pop(0)
sys.path.pop(0)