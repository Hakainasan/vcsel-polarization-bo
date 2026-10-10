"""Citation bookkeeping: every cited key exists, every entry states how it was verified, unverified references stay out."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_citations", ROOT / "scripts" / "check_citations.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def test_no_citation_problems():
    problems = cc.check(ROOT)
    assert problems == [], "\n".join(problems)
