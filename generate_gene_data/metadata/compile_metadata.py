import requests
import pandas as pd
from utilities.api_utilities import fetch_data
from tqdm import tqdm

"""
This requests only 
non failed data [failed$eqfalse]
coronal and sagittal data (plane_of_section ne 4)
non transgenic mice only (specimen(donor[transgenic_mouse_id$eqnull]))
ISH data only (treatments[name$eq%27ISH%27])
AntiSense probes only (probes[orientation_id$eq%272%27])
mice only (specimen(donor(organism[id$eq2])))
P56 only

"""

BASE_URL = "https://mouse.brain-map.org/api/v2/data/query.json"
CRITERIA = "criteria=model::SectionDataSet,rma::criteria,[plane_of_section_id$eq1],[failed$eqfalse],specimen(donor[transgenic_mouse_id$eqnull]),specimen(donor[disease_categories$eqnull]),specimen(donor(organism[id$eq2])),specimen(donor(age[days$eq56])),treatments[name$in%27NISSL%27,%27ISH%27]&include=plane_of_section,genes,treatments,specimen(donor(age)),probes,specimen(donor(organism)),specimen(donor(disease_categories))"


NUM_ROWS = 500
MAX_ROWS = None
api_data = fetch_data(BASE_URL, CRITERIA, NUM_ROWS, MAX_ROWS)
print(f"found {len(api_data)} experiments")
data = {
    "animal_id": [],
    "animal_name": [],
    "experiment_id": [],
    "sex": [],
    "gene": [],
    "age": [],
    "plane_of_section": [],
    "treatment": [],
    "probe_orientation": [],
    "sleep_state": [],
    "pixel_size": [],
}

for row in tqdm(api_data):
    animal_id_API = row["specimen_id"]
    animal_name = row["specimen"]["name"]
    experiment_id = row["id"]
    sex = row["specimen"]["donor"]["sex_full_name"]
    age = row["specimen"]["donor"]["age"]["name"]
    sleep_state = row["specimen"]["donor"]["sleep_state"]
    probe_orientation = (
        row["probes"][0]["orientation_id"] if len(row["probes"]) != 0 else "Nothing"
    )
    plane_of_section = row["plane_of_section"]["name"]
    if len(row["treatments"]) > 1:
        raise Exception("more than one treatment")
    if len(row["genes"]) > 1:
        raise Exception("more than one gene")
    treatment = row["treatments"][0]["name"]
    gene = row["genes"][0]["acronym"] if len(row["genes"]) != 0 else "Nothing"

    # Replace None with "None"
    animal_id_API = animal_id_API if animal_id_API is not None else "Nothing"
    animal_name = animal_name if animal_name is not None else "Nothing"
    experiment_id = experiment_id if experiment_id is not None else "Nothing"
    sex = sex if sex is not None else "Nothing"
    age = age if age is not None else "Nothing"
    sleep_state = sleep_state if sleep_state is not None else "Nothing"
    probe_orientation = (
        probe_orientation if probe_orientation is not None else "Nothing"
    )
    plane_of_section = plane_of_section if plane_of_section is not None else "Nothing"
    treatment = treatment if treatment is not None else "Nothing"
    gene = gene if gene is not None else "Nothing"

    data["animal_id"].append(animal_id_API)
    data["experiment_id"].append(experiment_id)
    data["sex"].append(sex)
    data["age"].append(age)
    data["plane_of_section"].append(plane_of_section)
    data["animal_name"].append(animal_name)
    data["gene"].append(gene)
    data["treatment"].append(treatment)
    data["sleep_state"].append(sleep_state)
    data["probe_orientation"].append(probe_orientation)

    # Retrieve section image metadata for this experiment.
    # We use the SectionDataSet endpoint (like in the second script) to include section_images.
    section_url = (
        f"http://api.brain-map.org/api/v2/data/SectionDataSet/{experiment_id}.json"
    )
    params = {"include": "section_images"}
    response = requests.get(section_url, params=params)
    if response.ok:
        section_data = response.json()
        # If section images exist, get the pixel size from the first image.
        if section_data.get("msg") and section_data["msg"][0].get("section_images"):
            first_image = section_data["msg"][0]["section_images"][0]
            # Assuming the JSON contains a 'pixel_size' attribute.
            pixel_size = first_image.get("resolution", "Undefined")
        else:
            pixel_size = "Undefined"
    else:
        pixel_size = "Undefined"

    data["pixel_size"].append(pixel_size)

metadata = pd.DataFrame(data)
"""
We checked which datasets were misnumbered and here we remove them. 
They were incorrectly numbered in the wrong direction
"""
exclude = [2193, 1878, 633]
metadata = metadata[~metadata["experiment_id"].isin(exclude)]
metadata = metadata.reset_index(drop=True)
metadata.to_csv("metadata.csv", index=None)
