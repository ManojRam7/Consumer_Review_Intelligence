"""Score new reviews with the best model from the last pipeline run.

    python predict.py "Beautifully written, I could not put it down."
    python predict.py "The plot dragged and the ending made no sense." "It was fine."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402

from review_intel import config as C  # noqa: E402
from review_intel.preprocess import TextCleaner  # noqa: E402


def main(texts: list[str]) -> None:
    model_path = C.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        sys.exit("No trained model found. Run `python run_pipeline.py` first.")
    model = joblib.load(model_path)
    cleaner = TextCleaner()
    cleaned = [cleaner.clean(t) for t in texts]
    for text, label in zip(texts, model.predict(cleaned)):
        print(f"{label:>8}  |  {text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
