import sys
from pathlib import Path

from Setup import *
model = get_model()

print(os.getcwd())

sys.path.insert(0, str(Path(__file__).parent.parent))

from Classification import classify_swing

from Overlay import overlay_swing_path

sys.path.pop(0)

