import os
import pandas as pd
from PyNutil.io.read_and_write import load_quint_json
from pathlib import Path

# load global metadata once
path = str(Path(__file__).parent.parent) + os.sep
metadata = pd.read_csv(Path(path, "metadata/metadata.csv"), index_col=None)


def name_from_id(experiment_id):
    return metadata["animal_name"][metadata["experiment_id"] == experiment_id].values[0]


def pixel_size_from_id(experiment_id):
    return metadata["pixel_size"][metadata["experiment_id"] == experiment_id].values[0]


def id_to_data_path(experiment_id, data_base):
    name = name_from_id(experiment_id)
    return os.path.join(data_base, str(name), str(experiment_id))


def id_to_quint_path(experiment_id, data_base):
    name = name_from_id(experiment_id)
    return os.path.join(data_base, "QUINT_registration_jsons", str(name)) + ".json"


def id_to_quint_json(experiment_id, data_base):
    path = id_to_quint_path(experiment_id, data_base)
    slices = load_quint_json(path, propagate_missing_values=False)["slices"]
    return [s for s in slices if s["filename"].split("/")[0] == str(experiment_id)]
