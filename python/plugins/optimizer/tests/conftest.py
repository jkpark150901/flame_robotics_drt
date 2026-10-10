from pathlib import Path
import sys


REPO_PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON_ROOT))
