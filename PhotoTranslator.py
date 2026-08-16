#!/usr/bin/env python3
"""Entry point. `python PhotoTranslator.py` or the frozen exe both land here."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pt.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
