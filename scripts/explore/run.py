"""Run one exploration script and save its output as evidence.

Usage, from the repo root:
    uv run python scripts/explore/run.py E01

Writes notes/evidence/E01.txt with a header recording the date, the code version
and the dataset version, then the script's output. Commit that file together with
the script, so a finding can always be traced back to the code and data that produced it.
"""
import datetime
import subprocess
import sys
from pathlib import Path

from common import ROOT, code_version, data_version


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    eid = sys.argv[1].upper()
    matches = sorted(Path(__file__).parent.glob(f"{eid}_*.py"))
    if len(matches) != 1:
        sys.exit(f"Expected exactly one script named {eid}_*.py in scripts/explore, found {len(matches)}.")
    script = matches[0]
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    header = [
        f"# Evidence {eid}",
        f"script:       scripts/explore/{script.name}",
        f"run at:       {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"memrecon:     {code_version()}",
        f"Memora data:  {data_version()}",
        f"exit status:  {proc.returncode}",
        "-" * 60,
    ]
    body = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    text = "\n".join(header) + "\n" + body
    out = ROOT / "notes" / "evidence" / f"{eid}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {out.relative_to(ROOT)}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
