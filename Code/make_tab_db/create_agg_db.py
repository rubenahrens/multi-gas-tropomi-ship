import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

from collections import defaultdict
from shapely.geometry import Polygon
import os
from pathlib import Path

import numpy as np
import glob

from joblib import Parallel, delayed
from tqdm import tqdm
import json

# import custom modules
import sys
sys.path.append('../')
sys.path.append('Code/')
from AIS import prep_ais
from HARP import harptools as ht

ROOT = "../../Data/"

def aggregate_ships(ais_df: pd.DataFrame, tile_center: Polygon):
    """
    Aggregates the ships present in the area of interest.
    Parameters:
    - ais_df: AIS data
    - tile_poly: polygon of the area of interest
    - time: time of the Tropomi overpass
    - NO2: NO2 data
    - SO2: SO2 data
    - HCHO: HCHO data
    - min_speed: minimum speed of the ships to consider
    - min_length: minimum length of the ship to consider
    - intp_ship_len: boolean indicating if the ship length should be interpolated
    - time_delta: time delta for the AIS data
    
    Returns:
    - ship_y: int binary number indicating if a ship has been present during two hours before the Tropomi overpass
    - n_ships: int number of ships present in the area
    - proxy_sum: float sum of the proxy values of the ships present in the area
    """
    # Create a GeoDataFrame from the AIS data
    ais_gdf = gpd.GeoDataFrame(ais_df, geometry='plume_track')
    
    # filter ais_gdf to only include ships that are within the tile
    ships_in_tile = ais_gdf[ais_gdf['plume_track'].intersects(tile_center)]
    
    # Get the ships that have been present during two hours before the Tropomi overpass
    ship_y = 0
    if not ships_in_tile.empty:
        ship_y = 1
    
    # Get the number of ships present in the area
    n_ships = ships_in_tile.shape[0]
    
    # Get the sum of the proxy values of the ships present in the area
    proxy_sum = sum(ships_in_tile['NOx_emission_proxy'])
        
    return ship_y, n_ships, proxy_sum


def aggregate_orbit_data(orbit_file:str, area_of_interest:list, ocean_poly:Polygon, aoi_name:str, Hparams:dict) -> pd.DataFrame:
    """
    Gets as input a TROPOMI orbit. Aligns AIS data to the orbit and returns the data.
    
    Parameters:
    - orbit_file (str): The path to the TROPOMI orbit file.
    - area_of_interest (list): The coordinates defining the area of interest [lon_min, lat_min, lon_max, lat_max].
    - ocean_poly (Polygon): The polygon representing the ocean area.
    - aoi_name (str): The name of the area of interest.
    - Hparams (dict): A dictionary containing various parameters for data processing.
    
    Returns:
    - pd.DataFrame: The aggregated data containing various gas measurements, ship information, and other parameters.
    """
   
    year, month, day = Path(orbit_file).parts[-4:-1]
    
    ais_df = prep_ais.get_ais_data(ROOT, orbit_file, aoi_name, area_of_interest, Hparams)
        
    if ais_df is None:
        return
    
    if Hparams['TILING'] == 'random':
        tiles, tile_centers, _ = ht.create_tiles_random(area_of_interest, square_size=80, ocean=ocean_poly)
    elif Hparams['TILING'] == 'grid':
        tiles, tile_centers, _ = ht.create_tiles_grid(area_of_interest, square_size=80)
    else:
        raise ValueError("Tiling strategy not implemented.")
    
    gas_data = defaultdict(list)
    
    for tile, tile_c in zip(tiles, tile_centers):
        
        grid, time = ht.import_product(orbit_file, tile, resolution=Hparams['RESOLUTION'])
        
        if grid == None or time == None:
            continue
        
        NO2 = grid['NO2_slant_column_number_density'].data[0,:,:]
        SO2 = grid['SO2_slant_column_number_density'].data[0,:,:]
        HCHO = grid['HCHO_slant_column_number_density'].data[0,:,:]
        
        # sparsity is the fraction of NaN values in the grid
        sparsity = [np.isnan(gas).sum() / gas.size for gas in [NO2, SO2, HCHO]]
        # if at least one of the gases has more than X% NaN values, skip the tile
        if max(sparsity) > Hparams['MAX_SPARSITY']:
            continue
        
        output = aggregate_ships(ais_df, tile_c)

        if output is None:
            continue
        
        ship_y, n_ships, proxy_sum = output
        
        for gas_name, gas in zip(["NO2", "SO2", "HCHO"], [NO2, SO2, HCHO]):
            gas_data[f'{gas_name}_scd_min'].append(np.nanmin(gas))
            gas_data[f'{gas_name}_scd_mean'].append(np.nanmean(gas))
            gas_data[f'{gas_name}_scd_median'].append(np.nanmedian(gas))
            gas_data[f'{gas_name}_scd_max'].append(np.nanmax(gas))
            gas_data[f'{gas_name}_scd_std'].append(np.nanstd(gas))

        # Create a dictionary for a single row
        gas_data['date'].append(f'{year}-{month}-{day}')
        gas_data['ship_y'].append(ship_y)
        gas_data['Proxy'].append(proxy_sum)
        gas_data['n_ships'].append(n_ships)
        # wind_mer_mean,wind_zon_mean,sensor_zenith_angle,solar_azimuth_angle,solar_zenith_angle
        gas_data['wind_mer_mean'].append(np.nanmean(grid['surface_meridional_wind_velocity'].data[0,:,:]))
        gas_data['wind_zon_mean'].append(np.nanmean(grid['surface_zonal_wind_velocity'].data[0,:,:]))
        gas_data['sensor_zenith_angle'].append(np.nanmean(grid['sensor_zenith_angle'].data[0,:,:]))
        gas_data['sensor_azimuth_angle'].append(np.nanmean(grid['sensor_zenith_angle'].data[0,:,:]))
        gas_data['solar_azimuth_angle'].append(np.nanmean(grid['solar_azimuth_angle'].data[0,:,:]))
        gas_data['solar_zenith_angle'].append(np.nanmean(grid['solar_zenith_angle'].data[0,:,:]))
        
        
    # return the data as a DataFrame
    return pd.DataFrame(gas_data)

if __name__ == "__main__":
    tropomi_files = glob.glob(ROOT + r"TinyEODATA/Mediterranean/Sentinel-5P/TROPOMI/L3__Merged_/*/*/*/*.nc")
    tropomi_files = [os.path.normpath(f) for f in tropomi_files]

    aoi_name = "Mediterranean"
    
    with open(ROOT+"aoi.json", "r") as f:
            aoi = json.load(f)
            
    area_of_interest = aoi[aoi_name]
    
    sea_polygon = ht.get_water_geometry(area_of_interest, ROOT)
    
    # same as used by kurchaba, for comparison
    Hparams = {
        'RESOLUTION': 0.005,
        'MIN_LENGTH': 90,
        'MIN_SPEED': 6,
        'INTERPOLATE_SHIP_LENGTH': True,
        'MAX_SPARSITY': 0.5,
        'ORBIT_TIME_METHOD': 'mean',
        'LOOKBACK_M' : 120,
        'TILING': 'random'
    }
        
    results = list(
        tqdm(
            Parallel(return_as="generator", n_jobs=-1)(
                delayed(aggregate_orbit_data)
                (orbit_file=orbit_file, area_of_interest=area_of_interest, ocean_poly=sea_polygon, aoi_name=aoi_name, Hparams=Hparams) 
                for orbit_file in tropomi_files
                ), 
            total=len(tropomi_files), 
            colour="green", 
            desc=f"Processing {aoi_name} data",
            leave=False
            )
    )
    
    all_data = pd.concat([result for result in results if result is not None])
    # shuffle the data
    all_data = all_data.sample(frac=1).reset_index(drop=True)
    
    # unique/incremental file name
    file_name = 0
    while glob.glob(ROOT+f"AGG_DB/{file_name}.csv"):
        file_name += 1    
    all_data.to_csv(ROOT+f"AGG_DB/{file_name}.csv", index=False)