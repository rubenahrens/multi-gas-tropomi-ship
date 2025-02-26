import requests
from shapely.geometry import Polygon, MultiPolygon, box

import json
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


# get keys from Data\keys.json
with open('../../Data/keys.json') as f:
    keys = json.load(f)
    # Define your parameters
    CLIENT_ID = keys["sentinelhub"]["CLIENT_ID"]
    CLIENT_SECRET = keys["sentinelhub"]["CLIENT_SECRET"]

# Create a session
client = BackendApplicationClient(client_id=CLIENT_ID)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
                          client_secret=CLIENT_SECRET, include_client_id=True)['access_token']

def request(date0, area_of_interest, product, date1=None, limit=100, absolute_orbit=None):
    """
    Return the Sentinel-5P products available for a given date and area of interest
    
    ----------
    Args:
        token (str): the API token
        date (str): the date in the format "YYYY-MM-DD"
        area_of_interest (list): the bounding box of the area of interest [minx, miny, maxx, maxy]
        product (str): the product type, e.g. "L2__NO2___"
        limit (int): the maximum number of products to return
        absolute_orbit (int): the absolute orbit number
    
    """
    if date1 is None:
        date1 = date0
    
    url = "https://creodias.sentinel-hub.com/api/v1/catalog/1.0.0/search"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
    }
    
    args = [
        {
            "op": "=",
            "args": [
            {
                "property": "s5p:timeliness"
            },
            "RPRO"
            ]
        }
        ]
    
    if product:
        args.append(
            {
            "op": "=",
            "args": [
            {
                "property": "s5p:type"
            },
            product
            ]
            }
        )
    
    if absolute_orbit:
        args.append(
            {
                "op": "=",
                "args": [
                {
                    "property": "sat:absolute_orbit"
                },
                absolute_orbit
                ]
            }
        )
    
    
    data = {
    "collections": [
        "sentinel-5p-l2"
    ],
    "datetime": f"{date0}T00:00:00Z/{date1}T23:59:59Z",
    "bbox": area_of_interest,
    "limit": limit,
    "filter": {
        "op": "and",
        "args": args
    },
    "filter-lang": "cql2-json"
    }
    response = requests.post(url, headers=headers, json=data).json()
    if response.get("code") == 401:
        raise ValueError(response.get("description"))
    return response


def calculate_coverage(polygon, area_of_interest):
    """
    Return the percentage of the area of interest that is covered by the polygon
    
    Args:
        polygon (shapely.geometry.Polygon): the polygon to be checked
        area_of_interest (list): the bounding box of the area of interest [lon_min, lat_min, lon_max, lat_max]
    
    Returns:
        float: the percentage of the area of interest that is covered by the polygon
    """
    assert type(polygon) == MultiPolygon or type(polygon) == Polygon
    
    # Convert the area of interest to a shapely Polygon using the bounding box coordinates
    area_of_interest_polygon = box(*area_of_interest)
    
    # Calculate the intersection area between the polygon and the area of interest
    intersection_area = polygon.intersection(area_of_interest_polygon).area
    
    # Calculate the area of the area of interest
    area_of_interest_area = area_of_interest_polygon.area
    
    # Calculate the coverage percentage
    coverage_percentage = (intersection_area / area_of_interest_area)
        
    assert type(coverage_percentage) == float
    
    return coverage_percentage

def parse_s5p_filename(filename, component: str = None):
    """
    Parses a Sentinel-5P product file name into its components.

    Parameters:
    filename (str): The Sentinel-5P product file name.

    Returns:
    dict: A dictionary with the components of the file name.
    """
    def parse_date(date_str):
        return date_str
    
    # Correctly extract components based on the specified indices and structure
    components = {
        "mission_name": filename[0:3],
        "processing_stream": filename[4:8],
        "product_identifier": filename[9:19],
        "start_of_granule": parse_date(filename[20:35]),
        "end_of_granule": parse_date(filename[36:51]),
        "orbit_id": int(filename[52:57]),
        "collection_number": int(filename[58:60]),
        "processor_version_number": int(filename[61:67]),
        "time_of_processing": parse_date(filename[68:83]),
    }
    
    if component:
        return components[component]
    else:
        return components

def valid_orbit(feature, collection_id, min_coverage, area_of_interest, gdf_orbit, i):
    filename = feature["id"]
          
    # Only consider orbits from the newest collection
    if filename[59] != collection_id:
        return False

    # Check if the orbit geometry covers min_coverage % of the area of interest
    coverage = calculate_coverage(gdf_orbit.iloc[i]["geometry"], area_of_interest)
    if coverage < min_coverage:
        return False

    return True