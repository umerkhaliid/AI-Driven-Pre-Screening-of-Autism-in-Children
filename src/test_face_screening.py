import importlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_dev_bypass_returns_autistic():
    with patch.dict(os.environ, {"DEV_BYPASS_FACE_SCREENING": "1"}):
        import src.face_screening as fs

        importlib.reload(fs)
        out = fs.predict_face_binary_or_bypass(None)
    assert out["is_autistic"] is True
    assert out["predicted_label"] == "Autistic"
    assert out.get("dev_bypass") is True


def test_resolve_path_none_when_default_missing():
    import src.config as cfg

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing_model.h5"
        os.environ.pop("FACE_CLASSIFIER_MODEL_PATH", None)
        with patch.object(cfg, "FACE_CLASSIFIER_DEFAULT_PATH", missing):
            import src.face_screening as fs

            importlib.reload(fs)
            assert fs.resolve_face_classifier_path() is None


if __name__ == "__main__":
    test_dev_bypass_returns_autistic()
    test_resolve_path_none_when_default_missing()
    print("ok")
