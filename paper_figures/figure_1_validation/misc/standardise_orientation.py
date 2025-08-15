"""
This script was used to standardie the left right orientation of all alignments.
It is not needed now as all alignments in this repo are standardised.
"""

import numpy as np
from glob import glob
import PyNutil
import json

atlas_width = 456 / 2
files = glob("raters/*/*.json")
files.extend(glob("raters/*/*/*.json"))
for fn in files:
    try:
        data = PyNutil.io.read_and_write.load_quint_json(
            fn, propagate_missing_values=False
        )
        slices = data["slices"]
        ux = [s["anchoring"][3] for s in slices]
    except Exception:
        data = PyNutil.io.read_and_write.load_quint_json(
            fn, propagate_missing_values=True
        )
        slices = data["slices"]
        ux = [s["anchoring"][3] for s in slices]
    signs = [np.sign(u) for u in ux]
    if not np.all(np.array(signs) == signs[0]):
        print(f"mismatched sign for {fn}")
    if signs[0] == -1:
        print(f"{fn} has negative sign – fixing and overwriting")
        for s in slices:
            s["anchoring"][0] = atlas_width - (s["anchoring"][0] - atlas_width)
            s["anchoring"][3] *= -1
            s["anchoring"][6] *= -1
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
