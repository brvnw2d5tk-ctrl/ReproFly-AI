"""Rebuild the five main figures from bundled frozen source data."""
from pathlib import Path
import os
import shutil
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
os.environ["REPROFLY_FIGURE_DATA_DIR"] = str(PACKAGE / 'paper/source_data')
from figure_v2.render_all import render_all

def main():
    output = PACKAGE / "output" / "figures"
    render_all(output)
    for number in range(1, 6):
        shutil.copy2(output / f"Figure{number}_v2.pdf",
                     output / f"Figure{number}.pdf")
        print("saved", f"Figure{number}")

if __name__ == "__main__":
    main()
