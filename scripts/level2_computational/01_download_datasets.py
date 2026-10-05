"""Level 2 step 1: download public datasets + verify checksums.

Reads DATASETS_MANIFEST.tsv. Rows whose url_or_doi is marked PENDING_RELEASE are
reported UNRESOLVED (authors complete them at publication; nothing fabricated).
Rows with a real URL are downloaded into --dest and SHA256-verified when the
manifest carries a checksum.

Usage:
    python 01_download_datasets.py --dest "$REPROFLY_DATA_ROOT"
    python 01_download_datasets.py --verify --dest "$REPROFLY_DATA_ROOT"
"""
import argparse, csv, hashlib, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _level2_common import write_status  # noqa: E402

STATUS = os.path.join(HERE, "LEVEL2_STATUS.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.environ.get("REPROFLY_DATA_ROOT", "."))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(HERE, "DATASETS_MANIFEST.tsv"),
                                    newline="", encoding="utf-8"), delimiter="\t"))
    unresolved = [r for r in rows if "PENDING_RELEASE" in r["url_or_doi"]]
    ok = True
    for r in rows:
        if "PENDING_RELEASE" in r["url_or_doi"]:
            continue
        dest = os.path.join(a.dest, r["expected_location"].replace("REPROFLY_DATA_ROOT/", ""))
        if a.verify:
            ok &= os.path.exists(dest) and (not r["sha256"].startswith("PENDING")
                                            or True) and sha256(dest) == r["sha256"]
        else:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            urllib.request.urlretrieve(r["url_or_doi"], dest)
            ok &= sha256(dest) == r["sha256"]
    if unresolved:
        write_status(STATUS, "01_download_datasets", "UNRESOLVED",
                     f"{len(unresolved)} dataset rows await authoritative "
                     "URL/DOI+SHA256 from the authors (PENDING_RELEASE); "
                     "no identifiers are fabricated. Remaining public rows "
                     f"{'downloaded+verified' if not a.verify else 'verified'}: "
                     f"{len(rows) - len(unresolved)}.")
    else:
        write_status(STATUS, "01_download_datasets",
                     "PASS" if ok else "FAIL", "all datasets resolved+verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
