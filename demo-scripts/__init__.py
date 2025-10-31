#!/usr/bin/env python3
from pathlib import Path
import sys, runpy

ROOT = Path(__file__).resolve().parent
# make 'utils' importable for all demos
sys.path.insert(0, str(ROOT / "utils"))
# make 'demo_scripts' runnable as a package
sys.path.insert(0, str(ROOT))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_demo.py <demo_module> [--args]")
        print("Examples:")
        print("  python run_demo.py demo_scripts.forecast_kit_dali --plot")
        sys.exit(2)
    mod = sys.argv[1]
    sys.argv = [mod] + sys.argv[2:]  # pass remaining args to the demo
    runpy.run_module(mod, run_name="__main__")

if __name__ == "__main__":
    main()
