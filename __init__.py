import sys
from pathlib import Path

from setup import *
model = get_model()

print(os.getcwd())

sys.path.insert(0, str(Path(__file__).parent.parent))

from classification import classify_swing

from overlay import overlay_swing_path

sys.path.pop(0)

