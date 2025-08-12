import json
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm

path_template = "/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/{}/{}/{}_metadata.json"
metadata = pd.read_csv("metadata/filtered_ISH.csv")
new_rows = []
for i, r in tqdm(metadata.iterrows(), total=len(metadata)):
    if r["treatment"] == "NISSL":
        continue
    else:
        json_path = path_template.format(
            r["animal_name"], r["experiment_id"], r["experiment_id"]
        )
    with open(json_path, "r") as rf:
        data = json.load(rf)

    if not np.allclose(
        [i["resolution"] for i in data[0]["section_images"]],
        data[0]["section_images"][0]["resolution"],
        atol=0.01,
    ):
        print("insconsistent resolution: ", r)
        break
    r["pixel_size"] = data[0]["section_images"][0]["resolution"]
    new_rows.append(r)
new_metadata = pd.DataFrame(new_rows)

for i, r in tqdm(metadata.iterrows(), total=len(metadata)):
    if r["treatment"] == "NISSL":
        search = new_metadata[new_metadata["animal_name"] == r["animal_name"]]
        if len(search) == 0:
            continue
        r["pixel_size"] = search["pixel_size"].values[0]
        new_rows.append(r)
new_metadata = pd.DataFrame(new_rows)


new_metadata.to_csv("filtered_ISH_pixel_size.csv", index=None)
