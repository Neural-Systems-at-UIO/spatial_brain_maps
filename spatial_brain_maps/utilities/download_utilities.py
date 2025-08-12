import requests
import os
import math
from io import BytesIO
from matplotlib.image import imread, imsave
import cv2


# the send_query function is taken from the ecallen package
def send_query(query_base, spec_id, args):
    response = requests.get(query_base.format(spec_id), params=args)
    if response.ok:
        json_tree = response.json()
        if json_tree["success"]:
            return json_tree
        else:
            exception_string = "did not complete api query successfully"
    else:
        exception_string = "API failure. Allen says: {}".format(response.reason)
    # raise an exception if the API request failed
    raise ValueError(exception_string)


def filter_metadata(metadata, column_name, filter_value):
    if filter_value != "ALL":
        if isinstance(filter_value, list):
            metadata = metadata[metadata[column_name].isin(filter_value)]
        else:
            filter_value = str(filter_value)
            metadata = metadata[metadata[column_name] == filter_value]
        if len(metadata) == 0:
            raise ValueError(
                f"No data found with the specified {column_name.upper()}(S)"
            )
    return metadata


def get_section_ids(experiment_id):
    response = send_query(
        "http://api.brain-map.org/api/v2/data/SectionDataSet/{}.json",
        experiment_id,
        {"include": "equalization,section_images"},
    )
    response = response["msg"]
    section_images = response[0]["section_images"]
    section_images = [i for i in section_images if i is not None]
    return section_images


def make_experiment_folder(output_dir, animal_name, experiment_id):
    output_path = os.path.join(output_dir, animal_name, str(experiment_id))
    os.makedirs(os.path.join(output_path, "10um"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "25um"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "expression"), exist_ok=True)


def double_n_times(value, n):
    for i in range(n):
        value *= 2
    return value


def downsample_image(image, source_pixel_size, target_pixel_size):
    downsampled_image = cv2.resize(
        image,
        (0, 0),
        fx=source_pixel_size / target_pixel_size,
        fy=source_pixel_size / target_pixel_size,
    )
    return downsampled_image


def save_image_to_disk(image, output_path):
    with open(output_path, "wb") as f:
        for chunk in image.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)


def download_and_save_image(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        save_image_to_disk(response, output_path)
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")


def download_image(
    output_dir,
    animal_name,
    experiment_id,
    image_id,
    image_number,
    resolution=None,
    view=None,
):
    """
    Downloads an image from the Allen Brain API and saves it to the specified directory.

    Parameters:
    - output_dir: Directory to save the downloaded image.
    - animal_name: Name of the animal.
    - experiment_id: ID of the experiment.
    - image_id: ID of the image.
    - image_number: Number of the image.
    - resolution: Resolution of the image (required if view is None).
    - view: View type, can be 'expression' or None.
    """
    allen_api = "http://api.brain-map.org/api/v2/image_download/"
    download_pattern = "{}{}?downsample={}&quality=100"

    if view == "expression":
        url = (
            download_pattern.format(allen_api, image_id, 0)
            + "&view=expression&filter=colormap&filterVals=0,1,0,256,0"
        )
        output_path = os.path.join(
            output_dir,
            animal_name,
            str(experiment_id),
            "expression",
            f"{image_id}_s{image_number:04}.jpg",
        )
        download_and_save_image(url, output_path)
    elif view is None:
        if resolution is None:
            raise ValueError("Resolution must be provided when view is None")

        downsample = math.floor(math.log2(10 / resolution))
        url = download_pattern.format(allen_api, image_id, downsample)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image_array = imread(image_data, format="jpeg")
        except requests.RequestException as e:
            print(f"Error downloading image: {e}")
            return
        except Exception as e:
            print(f"Error processing image: {e}")
            return

        source_pixel_size = double_n_times(resolution, downsample)
        for target_pixel_size in [10, 25]:
            downsampled_image = downsample_image(
                image_array, source_pixel_size, target_pixel_size
            )
            output_path = os.path.join(
                output_dir,
                animal_name,
                str(experiment_id),
                f"{target_pixel_size}um",
                f"{image_id}_s{image_number:04}.jpg",
            )
            imsave(output_path, downsampled_image)
