import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent))

from setup import *

print(os.getcwd())

model = get_model()

print(os.getcwd())

sys.path.insert(0, str(Path(__file__).parent / "yolov7"))

from classification import classify_swing

from overlay import overlay_swing_path

sys.path.pop(0)

