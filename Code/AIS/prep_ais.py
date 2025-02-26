import pandas as pd
import glob
import os
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
import harp
import geopandas as gpd
from pathlib import Path
from shapely.wkt import loads

def fill_values_ais(df, ship_db):
    # Fill missing values in IMO column based on MMSI in ship_db
    imo_to_mmsi = dict(zip(df['IMO'], df['MMSI']))
    mmsi_to_imo = dict(zip(df['MMSI']+ship_db['MMSI'], df['IMO']+ship_db['IMO']))
    callsign_to_mmsi = dict(zip(df['CallSign']+ship_db['CALL SIGN'], df['MMSI']+ship_db['MMSI']))
    name_to_mmsi = dict(zip(df['Name']+ship_db['VESSEL NAME'], df['MMSI']+ship_db['MMSI']))
    for index, row in df.iterrows():
        if pd.isna(row['IMO']):
            if row['MMSI'] in mmsi_to_imo:
                df.loc[index, 'IMO'] = mmsi_to_imo[row['MMSI']]  # Fill with the first matching IMO
        if pd.isna(row['MMSI']):
            if row['IMO'] in imo_to_mmsi:
                df.loc[index, 'MMSI'] = imo_to_mmsi[row['IMO']]  # Fill with the first matching MMSI
        if pd.isna(row['MMSI']):
            if row['CallSign'] in callsign_to_mmsi:
                df.loc[index, 'MMSI'] = callsign_to_mmsi[row['CallSign']]  # Fill with the first matching MMSI
        if pd.isna(row['MMSI']):
            if row['Name'] in name_to_mmsi:
                df.loc[index, 'MMSI'] = name_to_mmsi[row['Name']]
    return df

class AISdataLoader():
    """
    Class to load AIS data from csv files in the directory.
    
    Attributes:
    - data_dir (str): The directory path where the AIS data is located.
    - date (str): The date in the format 'YYYY-MM-DD'.
    - year (str): The year extracted from the date.
    - month (str): The month extracted from the date.
    - day (str): The day extracted from the date.
    - region (str): The name of the region.
    - region_code (str): The code corresponding to the region.
    - spatial_extent (list): The spatial extent of the region [min_longitude, min_latitude, max_longitude, max_latitude].
    - min_speed (int): The minimum speed threshold for filtering AIS data.
    - time_delta (tuple): The time window in hours for filtering AIS data.
    
    Methods:
    - wind_shift(ship_location, tropomi_data, time_delta, plot): Calculates the position of a ship's exhaust gasses based on the wind data, ship's location, and time difference.
    - load_ais_data(orbit_time): Loads AIS data from csv files within the specified time window and spatial extent.
    - load_ship_database(df): Loads ship database information and joins it with the AIS data.
    """
    
    ais_region_map = {
        'Mediterranean': 'EastMed', # should be 'EurMed'
        'Biscay Bay': 'EurMed',
        'Arabian Sea': 'ArabSea',
        'Bengal Bay': 'BengBay'
    }
    
    def __init__(self, data_dir:str, date:str, region:str, spatial_extent:list, Hparams:dict):
        """
        Initializes an instance of the AISdataLoader class.
        
        Args:
        - data_dir (str): The directory path where the AIS data is located.
        - date (str): The date in the format 'YYYY-MM-DD'.
        - region (str): The name of the region.
        - spatial_extent (list): The spatial extent of the region [min_longitude, min_latitude, max_longitude, max_latitude].
        - Hparams (dict): A dictionary containing various hyperparameters for the AIS data loader.
        
        Raises:
        - AssertionError: If the TROPOMI data does not exist for the specified date and region.
        """
        self.data_dir = data_dir
        self.date = date
        self.year = date[:4]
        self.month = date[5:7]
        self.day = date[8:]
        self.region = region
        self.region_code = self.ais_region_map[region]
        self.spatial_extent = spatial_extent
        self.min_speed = Hparams["MIN_SPEED"]
        self.time_delta = np.timedelta64(Hparams["LOOKBACK_M"], 'm')
        self.time_buffer = np.timedelta64(30, 'm')
        self.min_ship_length = Hparams["MIN_LENGTH"]
        self.interpolate_ship_len = Hparams["INTERPOLATE_SHIP_LENGTH"]
        # check if the merged tropomi data exists
        assert os.path.exists(f'{data_dir}/TinyEODATA/{region}/Sentinel-5P/TROPOMI/L3__Merged_/{self.year}/{self.month}/{self.day}'), \
        f"TROPOMI data does not exist for date {date} and region {region}"
        
    def merge_ship_info(self, ais:pd.DataFrame, db:pd.DataFrame) -> pd.DataFrame:
        """
        Merges ship information from AIS and shipDB.
        Merge is done in the following order: MMSI, IMO, Name, CallSign.
        If a column is not found in shipDB, the next column is used.
        If no column is found, the row is dropped.
        
        Inspired by method of Solomiia Kurchaba, Artur Sokolovsky, Jasper van Vliet, Fons J. Verbeek, Cor J. Veenman,
        analysis for the detection of NO2 plumes from seagoing ships using TROPOMI data,

        Args:
            ais (pd.DataFrame): DataFrame containing AIS data.
            db (pd.DataFrame): DataFrame containing ship information database.

        Returns:
            pd.DataFrame: Merged DataFrame with prioritized information.
        """
        well_merged = []
        to_merge = ais.copy()
        for merge_col_x, merge_col_y in zip(['MMSI', 'IMO', 'CallSign'], ['MMSI', 'IMO', 'CALL SIGN']):
            merged = to_merge.merge(db.dropna(subset=[merge_col_y]), how='left', left_on=merge_col_x, right_on=merge_col_y, suffixes=('', '_SDB'))
            to_merge = merged[merged['LENGTH REGISTERED'].isna()][ais.columns]
            merged = merged[~merged['LENGTH REGISTERED'].isna()][list(ais.columns)+['LENGTH REGISTERED']+['AGG VESSEL TYPE']+['TIER']]
            well_merged.append(merged)
            del merged
            if to_merge.empty:
                break
        return pd.concat(well_merged)
    
    def wind_shift(self, ship_location:shp.Point, tropomi_data:harp.Product, orbit_time:np.datetime64, 
                   point_time:np.datetime64, plot:bool = False, exclude_bounds:bool = False) -> shp.Point:
        """
        wind_shift calculates the position of a ship's exhaust gasses based on the wind data, the ship's location and the time difference.

        Zonal and meridional flow are directions and regions of fluid flow on a globe. 
        Zonal flow follows a pattern along latitudinal lines, latitudinal circles or in the west-east direction.

        Meridional flow follows a pattern from north to south, or from south to north, along the Earth's longitude lines, 
        longitudinal circles (meridian) or in the north-south direction.
        
        The main idea of this is to find the nearest point in the array to the given point and then check if the point is within the bounds.
        If the point is not within the bounds, then the function returns None.
        
        This function has been inspired by the following paper:
        Solomiia Kurchaba, Artur Sokolovsky, Jasper van Vliet, Fons J. Verbeek, Cor J. Veenman,
        Sensitivity analysis for the detection of NO2 plumes from seagoing ships using TROPOMI data,

        Args:
        ship_location: shapely Point, location of the ship [longitude, latitude]
        tropomi_data: xarray dataset, TROPOMI data. Must contain zona and meridional wind data
        orbit_time: np.datetime64, time of the orbit overpass
        point_time: np.datetime64, time of the ship location
                
        Optional Args:
        plot: bool, plot the ship and plume origin
        exclude_bounds: bool, return None if the plume or ship origin is out of bounds of the TROPOMI data
        
        Returns:
        shapely Point: new location of the ship
        """
        assert orbit_time >= point_time, "Orbit time must be greater than point time"
        time_delta = orbit_time - point_time
        
        lat_bounds = tropomi_data['latitude_bounds'].data
        lon_bounds = tropomi_data['longitude_bounds'].data
        latitudes = tropomi_data['latitude'].data
        longitudes = tropomi_data['longitude'].data
        time_seconds = time_delta.astype('timedelta64[s]').astype(int)
        
        if time_seconds == 0:
            return ship_location

        def find_nearest_point_index(point):
            """
            Find the index of the nearest point in the array to the given point
            """
            lon_diff = np.abs(longitudes - point.x)
            lat_diff = np.abs(latitudes - point.y)
            rel_dist = lat_diff + lon_diff
            return np.argmin(rel_dist)
        
        def within_bounds(point, idx):
            """
            Check if the point is within the bounds
            """
            box = zip(lon_bounds[idx], lat_bounds[idx])
            box = shp.geometry.Polygon(box)
            return point.within(box)

        ship_idx = find_nearest_point_index(ship_location)
        if exclude_bounds and not within_bounds(ship_location, ship_idx):
            return None
        
        zonal_wind = tropomi_data['surface_zonal_wind_velocity'].data[ship_idx]
        meridional_wind = tropomi_data['surface_meridional_wind_velocity'].data[ship_idx]
        
        # latitude and longitude degrees are approximately 111.139 km, 
        # assuming a perferctly spherical earth
        degree_in_meters = 111139
        
        # longitude shift in meters along the zonal axis
        lon_shift =  zonal_wind * time_seconds
        # convert the longitude shift to degrees
        lon_shift = lon_shift / degree_in_meters
        
        # latitude shift in meters along the meridional axis
        lat_shift = meridional_wind * time_seconds
        # convert the latitude shift to degrees
        lat_shift = lat_shift / degree_in_meters
        
        plume_origin = shp.geometry.Point(ship_location.x + lon_shift, 
                                          ship_location.y + lat_shift)
        
        # check if the plume origin is within the bounds
        plume_idx = find_nearest_point_index(plume_origin)
        
        if exclude_bounds and not within_bounds(plume_origin, plume_idx):
            return None
        
        if plot:
            plt.plot(lon_bounds[ship_idx], lat_bounds[ship_idx])
            plt.plot(ship_location.x, ship_location.y, 'ro')
            plt.plot(plume_origin.x, plume_origin.y, 'bo')
            plt.legend(['Bounds', 'Ship origin', 'Plume origin'])
            plt.show()
        
        return plume_origin
        
    
    def load_ais_data(self, orbit_time:np.datetime64) -> pd.DataFrame:
        """
        Load AIS data from csv files in the directory
        
        Args:
        orbit_time: np.datetime64, time of the orbit overpass
                
        Returns:
        df: pandas DataFrame, concatenated dataframe of AIS data
        This dataframe is filtered on spatial extent and time window
        """
        abs_data_dir = os.path.abspath(self.data_dir)
        
        orbit_time_start = orbit_time - self.time_delta - self.time_buffer # buffer of 30 minutes
        orbit_time_end = orbit_time + self.time_buffer # buffer of 30 minutes
                
        # check if time window overlaps with a different day
        orbit_dates = pd.to_datetime([orbit_time_start, orbit_time_end])
        orbit_dates = [date.strftime('%Y%m%d') for date in orbit_dates]
        files = []

        for date in set(orbit_dates):
            # get list of all csv files in the directory with glob
            files.extend(glob.glob(f'{abs_data_dir}/AIS/AIS_raw/{self.region}/ShipTrack_{self.region_code}_{date}Tile*.csv'))
        
        fp = []
        for file in files:
            tile_df = pd.read_csv(file, usecols=['MMSI', 'latitude', 'longitude', 'speed', 'timestamp', 'IMO', 'Name', 'CallSign'])
            if len(tile_df) > 0:
                fp.append(tile_df)
                
        df = pd.concat(fp)

        # convert timestamp to <class 'np.datetime64'>
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%SZ')
        
        # filter on spatial extent, add 1 degree to the spatial extent to account for the buffer
        df = df[(df['latitude'] > self.spatial_extent[1] - 1) & (df['latitude'] < self.spatial_extent[3] + 1)]
        df = df[(df['longitude'] > self.spatial_extent[0] - 1) & (df['longitude'] < self.spatial_extent[2] + 1)]
        
        df = df[(df['timestamp'] > orbit_time_start) & (df['timestamp'] < orbit_time_end)]
        
        # assign all lower caps to Name column
        df['Name'] = df['Name'].str.lower()

        # load ship database
        df = self.load_ship_database(df)
        
        return df
        
    def load_ship_database(self, df):
        """
        Load ship database from csv file in the directory
        
        Args:
        df: DataFrame, AIS data
        
        Returns:
        DataFrame: AIS data with ship database information joined
        
        """
        ship_db = pd.read_csv(f'{self.data_dir}/AIS/ShipDB/Vessels4AVES.csv', 
                              usecols=['MMSI','YEAR OF BUILD', 'DATE OF BUILD', 'LENGTH REGISTERED',
                                       'AGG VESSEL TYPE', 'GT', 'VESSEL NAME', 'CALL SIGN', 'IMO', 'TIER'])
        
        # convert the ship names to lower case
        ship_db["VESSEL NAME"] = ship_db["VESSEL NAME"].str.lower()
        
        # remove ships with call sign 'TEST'
        ship_db = ship_db[ship_db['CALL SIGN'] != 'TEST']
                        
        # Filter on ship types
        ship_db = ship_db[ship_db['AGG VESSEL TYPE'].isin(['Tanker', 'Bulk', 'Container', 'Passenger'])]
        
        def GT_to_len(x):
            # source: [FleetTierStudy.ipynb]
            return 8.494e-14*x**3 - 3.223e-08*x**2 + 0.004532*x + 84.66
        
        # For ships with missing length, fill with the length calculated from GT
        ship_db['LENGTH REGISTERED'] = ship_db['LENGTH REGISTERED'].fillna(GT_to_len(ship_db['GT']))
        
        # For remaining ships with missing length, fill with the mean length of the ship type and tier
        ship_db['LENGTH REGISTERED'] = ship_db['LENGTH REGISTERED'].fillna(ship_db.groupby(['AGG VESSEL TYPE', 'TIER'])['LENGTH REGISTERED'].transform('mean'))
        
        def len_to_GT(x):
            # source derivation: [FleetTierStudy.ipynb]
            return 0.000539*x**3 + 1.577*x**2 - 219.5*x + 8904
        
        # For ships with missing GT, fill with the GT calculated from length
        ship_db['GT'] = ship_db['GT'].fillna(len_to_GT(ship_db['LENGTH REGISTERED']))
                
        # filter on the length of the ship, if the length of the ship is less than 90 meters, drop the row
        ship_db = ship_db[ship_db['LENGTH REGISTERED'] > self.min_ship_length]
        
        # TODO: check
        # filter on the date of build, if the date of build is after the date of the orbit, drop the row
        ship_db['DATE OF BUILD'] = ship_db['DATE OF BUILD'].fillna(self.date)
        ship_db = ship_db[ship_db['DATE OF BUILD'] <= self.date]

        # sort shipdb by 'YEAR OF BUILD' in descending order
        ship_db = ship_db.sort_values('YEAR OF BUILD', ascending=False)

        # if there are duplicate MMSI values, keep the newest ship based on 'YEAR OF BUILD'
        ship_db = ship_db[~ship_db.duplicated('MMSI', keep='first')] # this can

        # join the two dataframes on the MMSI column
        df = self.merge_ship_info(df, ship_db)
        
        # check if nan values are present in LENGTH REGISTERED column
        assert df['LENGTH REGISTERED'].isnull().sum() == 0, "LENGTH REGISTERED column contains missing values"
        
        df = fill_values_ais(df, ship_db)
        
        # sort by Ship_ID and timestamp
        df.sort_values(by=['MMSI', 'timestamp'], inplace=True)
        return df
        
    def interpolate_ship_location(self, ship_df, time, cap):
        """
        Interpolates the location of the ship at a given time based on ship data.

        Parameters:
        ship_df (DataFrame): The DataFrame containing ship data.
        time (datetime): The time at which to interpolate the ship location.
        cap (str): Whether to use the ceiling or floor value if there are no points before and after the time.

        Returns:
        Point: The interpolated ship location as a shapely Point object.
        """
        assert cap in ['ceiling', 'floor', None], "cap must be either 'ceiling' or 'floor' or None"
        # check if the dataframe is sorted by timestamp
        assert ship_df['timestamp'].is_monotonic_increasing, "DataFrame is not sorted by timestamp"
        # interpolate the location of the ship at the time of the orbit
        before_df = ship_df[ship_df['timestamp'] <= time]
        after_df = ship_df[ship_df['timestamp'] >= time]
        
        if len(before_df) == 0 and len(after_df) == 0:
            return None
        elif len(before_df) == 0:        
            if cap == 'floor':
                after_df = after_df[after_df['timestamp'] <= time + self.time_delta]
                if len(after_df) == 0:
                    return None
                return shp.geometry.Point(after_df['longitude'].iloc[0], after_df['latitude'].iloc[0])
            else:
                return None
        elif len(after_df) == 0:
            if cap == 'ceiling':
                before_df = before_df[before_df['timestamp'] >= time - self.time_delta]
                if len(before_df) == 0:
                    return None
                return shp.geometry.Point(before_df['longitude'].iloc[-1], before_df['latitude'].iloc[-1])
            else:
                return None
        
        point_a = before_df.iloc[-1]
        point_b = after_df.iloc[0]
        if point_a['timestamp'] == point_b['timestamp']:
            return shp.geometry.Point(point_a['longitude'], point_a['latitude'])
                
        # calculate the fraction of the time between the two points
        fraction = (time - point_a['timestamp']) / (point_b['timestamp'] - point_a['timestamp'])
        
        # create shapely line objects
        line = shp.geometry.LineString([(point_a['longitude'], point_a['latitude']), (point_b['longitude'], point_b['latitude'])])
        
        # interpolate the location of the ship at the time of the orbit
        return line.interpolate(fraction, normalized=True)


    def load_tropomi_data(self, orbit_time_method : str, plot : bool = False):
        """
        Load TROPOMI data from netcdf files in the directory
        
        Args:
        orbit_time_method: str, method to calculate the time of the orbit
        plot: bool, plot the ship and plume origin
        """
        # filter on the area of interest
        operations = ";".join([
            f"latitude>{self.spatial_extent[1]};latitude<{self.spatial_extent[3]}",
            f"longitude>{self.spatial_extent[0]};longitude<{self.spatial_extent[2]}",
            "keep(latitude_bounds,longitude_bounds,latitude,longitude,surface_zonal_wind_velocity, \
                    surface_meridional_wind_velocity,datetime_start,datetime_length)",
        ])

        tropomi_data = harp.import_product(f'{self.data_dir}/TinyEODATA/{self.region}/Sentinel-5P/TROPOMI/L3__Merged_/{self.year}/{self.month}/{self.day}/*.nc',
                                           operations=operations)
        
        seconds_since_ref = tropomi_data["datetime_start"].data
        time_length = tropomi_data["datetime_length"].data
        
        if orbit_time_method == 'mean':
            # calculate the mean time of the orbit
            orbit_time_mean = np.mean(seconds_since_ref) + np.mean(time_length) / 2
        elif orbit_time_method == 'max':
            # calculate the maximum time of the orbit
            orbit_time_mean = np.max(seconds_since_ref) + np.max(time_length) / 2
        elif orbit_time_method == 'min':
            # calculate the minimum time of the orbit
            orbit_time_mean = np.min(seconds_since_ref) + np.min(time_length) / 2
        else:
            raise ValueError("orbit_time_method must be either 'mean', 'max' or 'min'")
        
        # Convert seconds since 2010-01-01 to np.datetime64 timestamp
        orbit_time_mean = np.datetime64('2010-01-01T00:00:00') + np.timedelta64(int(orbit_time_mean), 's')
                    
        # load AIS data
        ais_df = self.load_ais_data(orbit_time_mean)
        
        output = {'MMSI': [], 'ship_track': [], 'plume_track': [], 'NOx_emission_proxy': [], 'timestamp': []}

        # for each ship, find the location of the ship at the time of the orbit
        # find two points before and after the orbit time
        for ship_id in ais_df['MMSI'].unique():
            ship_df = ais_df[ais_df['MMSI'] == ship_id]
            
            approx_speed = ship_df['speed'].mean()
                
            # filter out ships with an average speed of less than 5 knots
            if approx_speed < self.min_speed:
                continue
                        
            points_in_range = ship_df[ship_df['timestamp'].between(orbit_time_mean - self.time_delta, orbit_time_mean)]
            
            if len(points_in_range) == 0:
                continue
            
            ship_track = points_in_range[['longitude', 'latitude']].values
            track_time = points_in_range['timestamp'].values
                                    
            # interpolate the location of the ship 2 hours before the orbit
            start = self.interpolate_ship_location(ship_df, orbit_time_mean - self.time_delta, cap=None)
            if start:
                # add start as the first element of line
                ship_track = np.vstack([[start.x,start.y], ship_track])
                track_time = np.hstack([orbit_time_mean-self.time_delta, track_time])                
            
            
            plume_track = []
            for point, delta_t in zip(ship_track, track_time):
                point = shp.geometry.Point(point)
                # shift the ship_location_before point based on wind data
                plume_track.append(self.wind_shift(point, tropomi_data, orbit_time_mean, delta_t, plot=False))
                
            # interpolate the location of the ship at the time of the orbit
            finish = self.interpolate_ship_location(ship_df, orbit_time_mean, cap=None)
            if finish:
                # add finish as the last element of line
                ship_track = np.vstack([ship_track, [finish.x,finish.y]])
                plume_track.append(finish)
            
            if len(plume_track) < 2 or len(ship_track) < 2:
                continue
            ship_track = shp.geometry.LineString(ship_track)
            plume_track = shp.geometry.LineString(plume_track)
                        
            # add the line to a dataframe with the MMSI
            output['MMSI'].append(ship_id)
            output['ship_track'].append(ship_track)
            output['plume_track'].append(plume_track)
            output['timestamp'].append(orbit_time_mean)
            # Add proxy data https://doi.org/10.1088/1748-9326/abc445
            assert len(ship_df['LENGTH REGISTERED'].unique()) == 1, f"Multiple ship lengths found {ship_df['Name']}"
            ship_length = ship_df['LENGTH REGISTERED'].iloc[0]
            knot_to_m_s = 0.514444
            output['NOx_emission_proxy'].append(ship_length**2 * (approx_speed*knot_to_m_s)**3)
                        
        output = gpd.GeoDataFrame(output)
        
        if plot:
            plot_gdf = output.copy()
            plot_gdf.set_geometry('ship_track', inplace=True)
            plt.close()
            fig, ax = plt.subplots(figsize=(10,10))
            plot_gdf.plot(ax=ax, color='blue', alpha=0.5)
            plot_gdf.set_geometry('plume_track', inplace=True)
            plot_gdf.plot(ax=ax, color='red', alpha=0.5)
            plt.savefig("tmp.png")
        
        return output
    
    
def get_ais_data(ROOT, orbit_file, aoi_name, area_of_interest, Hparams):
    """
    Get AIS data from the directory
    
    Args:
    ROOT: str, root directory of the data
    orbit_file: pathlib.Path, path to the TROPOMI data
    aoi_name: str, name of the area of interest
    area_of_interest: list, spatial extent of the area of interest
    Hparams: dict, hyperparameters for the AIS data loader
    
    Returns:
    ais_df: pd.DataFrame, AIS data with the ship's location shifted by the wind data
    """
    year, month, day = Path(orbit_file).parts[-4:-1]
        
    # check if AIS data is available for the given date
    if len(glob.glob(ROOT+f"AIS/AIS_raw/{aoi_name}/*{year}{month}{day}*.csv")) == 0:
        return
    
    filename = "_".join([str(Hparams[key]) for key in ["MIN_SPEED", "LOOKBACK_M", "MIN_LENGTH", "INTERPOLATE_SHIP_LENGTH", "ORBIT_TIME_METHOD"]])
    path = f"{ROOT}AIS/AIS_processed/{aoi_name}/{year}/{month}/{day}/{filename}.csv"
    
    if os.path.exists(path):
        ais_df = pd.read_csv(path)
        ais_df['ship_track'] = ais_df['ship_track'].apply(loads)
        ais_df['plume_track'] = ais_df['plume_track'].apply(loads)
        return ais_df
    
    AIS_ldr = AISdataLoader(ROOT, f"{year}-{month}-{day}", aoi_name, area_of_interest, Hparams)
    ais_df = AIS_ldr.load_tropomi_data(Hparams["ORBIT_TIME_METHOD"])
    ais_df.to_csv(path, index=False)
    # print(ais_df)
    return ais_df