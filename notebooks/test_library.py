from pathlib import Path
import sys

# Allow running this file directly from `notebooks/` while importing project code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This should work from any directory
from pangaea_readiness import evaluate

result = evaluate("10.1594/PANGAEA.787140", use_ai=True)
print(result.summary())
print(result.score)
print(result.flag)
print(result.ai_domain)