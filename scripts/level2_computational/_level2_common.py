"""Shared Level-2 status helpers."""
import json, os

def write_status(path, step, status, detail):
    entries = []
    if os.path.exists(path):
        try:
            entries = json.load(open(path, encoding="utf-8"))
        except Exception:
            entries = []
    entries = [e for e in entries if e.get("step") != step]
    entries.append({"step": step, "status": status, "detail": detail})
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, indent=1); f.write("\n")
    print(f"[{status}] {step}: {detail}")
    return status == "PASS"
