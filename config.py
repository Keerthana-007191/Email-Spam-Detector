from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "spam_pipeline.joblib"

RANDOM_STATE = 42

# ham = 0, spam = 1  (spam is the positive class)
LABEL_MAP = {"ham": 0, "spam": 1}
LABEL_NAMES = {0: "Legitimate", 1: "Spam"}

TEXT_COLUMN_CANDIDATES = ["message", "text", "body", "email", "content", "email_text"]
SUBJECT_COLUMN_CANDIDATES = ["subject", "title"]
LABEL_COLUMN_CANDIDATES = ["spam/ham", "label", "class", "category", "spam", "target", "v1"]