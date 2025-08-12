import requests
import time


def fetch_data(base_url, criteria, num_rows, max_retries=10):
    start_row = 0
    all_data = []

    while True:
        request_str = f"{base_url}?{criteria}&num_rows={num_rows}&start_row={start_row}"
        retries = 0

        while retries < max_retries:
            response = requests.get(request_str)
            if response.status_code == 200:
                data = response.json()["msg"]
                if not data:
                    return all_data
                all_data.extend(data)
                start_row += num_rows
                break
            else:
                retries += 1
                time.sleep(1)  # Wait for 1 second before retrying
                if retries == max_retries:
                    raise Exception(
                        f"Request failed after {max_retries} retries with status code {response.status_code}: {response.text}"
                    )
    return all_data
