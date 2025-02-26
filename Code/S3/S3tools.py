import boto3
import os
import json

# get keys from Data\keys.json
with open('../../Data/keys.json') as f:
    keys = json.load(f)
    # Define your parameters
    access_key = keys["s3"]["access_key"]
    secret_key = keys["s3"]["secret_key"]

session = boto3.session.Session()
s3 = boto3.resource(
    's3',
    endpoint_url='https://eodata.dataspace.copernicus.eu',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='default'
)  # generated secrets

def download(product: str, target: str = "") -> None:
    """
    Downloads every file in bucket with provided product as prefix

    Raises FileNotFoundError if the product was not found

    Args:
        bucket: boto3 Resource bucket object
        product: Path to product
        target: Local catalog for downloaded files. Should end with an `/`. Default current directory.
    """
    
    bucket = s3.Bucket("eodata")
    
    # check if the file already exists
    if os.path.isfile(f"{target}{product}"):
        return
    
    # TODO: check if the merged file already exists

    files = bucket.objects.filter(Prefix=product)
    if not list(files):
        raise FileNotFoundError(f"Could not find any files for {product}")

    for file in files:
        file_path = f"{target}{file.key}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.isdir(file_path):
            bucket.download_file(file.key, file_path)

if __name__ == "__main__":
    # path to the product to download
    download("Sentinel-5P/TROPOMI/L2__NO2___/2019/06/14/S5P_RPRO_L2__NO2____20190614T104316_20190614T122446_08641_03_020400_20221105T175851/S5P_RPRO_L2__NO2____20190614T104316_20190614T122446_08641_03_020400_20221105T175851.nc")