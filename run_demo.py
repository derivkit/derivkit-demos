"""Utility to run demo scripts with proper module paths."""

from __future__ import annotations
from pathlib import Path
import argparse, runpy, sys, re

ROOT  = Path(__file__).resolve().parent
DEMOS = ROOT / "demo-scripts"
UTILS = ROOT / "utils"

def _norm(s: str) -> str:
    """Normalizes a demo name for fuzzy matching."""
    s = s.lower()
    s = s[:-3] if s.endswith(".py") else s
    return re.sub(r"[^a-z0-9]+", "", s)

def _catalog():
    """Catalogs available demo scripts."""
    entries = []
    for p in sorted(DEMOS.glob("*.py")):
        if p.name == "__init__.py":
            continue
        stem = p.stem
        # grab leading numeric id if present (e.g., "07" from "07-foo-bar")
        m = re.match(r"^(\d+)[-_]?", stem)
        demo_id = m.group(1) if m else ""
        entries.append({
            "path": p,
            "name": p.name,
            "stem": stem,
            "id": demo_id,
            "norm": _norm(p.name),
        })
    return entries

def list_demos():
    """Prints available demos to stdout."""
    entries = _catalog()
    width = max((len(e["name"]) for e in entries), default=10)
    print("Available demos:\n")
    for e in entries:
        tag = f"[{e['id']}]" if e["id"] else "   "
        print(f"  {tag:>5}  {e['name']:<{width}}   ({e['path'].relative_to(ROOT)})")

def resolve_demo(query: str):
    """Resolves a demo script from a query string. """
    entries = _catalog()
    if not entries:
        raise SystemExit("No demos found under demo-scripts/")

    qnorm = _norm(query)
    # 1) exact filename or stem
    for key in ("name", "stem"):
        for e in entries:
            if query == e[key]:
                return e
    # 2) numeric id
    if query.isdigit():
        cand = [e for e in entries if e["id"] == query]
        if len(cand) == 1:
            return cand[0]
        if len(cand) > 1:
            names = ", ".join(c["name"] for c in cand)
            raise SystemExit(f"Ambiguous id '{query}': {names}")
    # 3) normalized (handles dashed names, partials)
    cand = [e for e in entries if qnorm and e["norm"].startswith(qnorm)]
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        names = ", ".join(c["name"] for c in cand)
        raise SystemExit(f"Ambiguous name '{query}': {names}")
    raise SystemExit(f"Demo '{query}' not found. Run: python run_demo.py --list")

def main():
    """Main entry point for running a demo script."""
    ap = argparse.ArgumentParser(description="Run DerivKit demos without installing.")
    ap.add_argument("demo", nargs="?", help="demo selector (id, filename, or fuzzy name)")
    ap.add_argument("demo_args", nargs=argparse.REMAINDER,
                    help="arguments passed to the demo (prefix with -- if needed)")
    ap.add_argument("--list", action="store_true", help="list available demos and exit")
    ns = ap.parse_args()

    if ns.list or not ns.demo:
        list_demos()
        if not ns.list:
            print("\nUsage:\n  python run_demo.py 07 -- --plot\n  python run_demo.py dali -- --plot")
        return

    entry = resolve_demo(ns.demo)

    # ensure demos can do: from utils.style import apply_plot_style
    sys.path.insert(0, str(UTILS))
    # also let demos import sibling modules if they ever do
    sys.path.insert(0, str(ROOT))

    # pass through the remaining args to the demo
    sys.argv = [entry["name"]] + ns.demo_args
    runpy.run_path(str(entry["path"]), run_name="__main__")

if __name__ == "__main__":
    main()
