#
# requires:
# pip install s3fs 'xarray[complete]' matplotlib
#

import s3fs
import xarray as xr
import matplotlib.pyplot as plt

# --- Step 1: Connect to NOAA's GOES16 S3 bucket ---
fs = s3fs.S3FileSystem(anon=True)

# Choose product and date
product = 'ABI-L2-FDCF'   # Fire/Hot Spot Characterization
year = '2024'
day_of_year = '312'       # e.g. November 8 = 312th day of year
hour = '18'

# Path pattern in S3 bucket
path = f'noaa-goes16/{product}/{year}/{day_of_year}/{hour}/'

# List files available in that hour
files = fs.ls(path)
print(f"Found {len(files)} files")

# --- Step 2: Pick one file ---
file = 's3://' + files[0]
print("Opening file:", file)

# --- Step 3: Open with xarray ---
ds = xr.open_dataset(fs.open(file))

print(ds)

# --- Step 4: Visualize a variable, e.g., fire mask or brightness temperature ---
# Fire mask: 0 = no fire, 10 = fire
if 'Mask' in ds.variables:
    plt.imshow(ds['Mask'], cmap='hot')
    plt.title('GOES-16 Fire Mask')
    plt.colorbar()
    plt.show()
elif 'Temp' in ds.variables:
    plt.imshow(ds['Temp'], cmap='inferno')
    plt.title('GOES-16 Fire Temperature (K)')
    plt.colorbar()
    plt.show()
