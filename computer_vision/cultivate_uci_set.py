import os
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import tarfile

url = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
dataset_dir = "/Users/abhi/Desktop/FERDA/UCI_FIRE"

tar_links = soup.find_all('a', href=lambda href: href and href.endswith('.tgz'))

print(tar_links)

for link in tar_links:
    file_url = urljoin(url, link['href'])  # Handle relative URLs
    filename = os.path.basename(file_url)
    filepath = os.path.join(dataset_dir, filename)


    print(f"Downloading {file_url} to {filepath}...")

    try:
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded: {filename}")
     

    except Exception as e:
        print(f"Failed to download {file_url}: {e}")

    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(dataset_dir)

        print(f"Extracted: {filename}")
        os.remove(filepath)

            
    except:

        print(f"Unable to extract: {filename}")

    



