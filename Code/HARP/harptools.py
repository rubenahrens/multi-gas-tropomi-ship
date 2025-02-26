import numpy as np
import shapely as shp
import os
import harp
from shapely.geometry import box, MultiPolygon
import geopandas as gpd
from geopandas import read_file
import matplotlib.pyplot as plt

def get_water_geometry(area_of_interest: list, root: str, simplify=True) -> MultiPolygon:
    """
    Retrieves the water geometry within a specified area of interest.

    Parameters:
    - area_of_interest (list): The area of interest in the form [lon_min, lat_min, lon_max, lat_max].
    - root (str): The root directory path.

    Returns:
    - MultiPolygon: A simplified MultiPolygon object representing the water geometry within the area of interest.
    """
    if type(area_of_interest) == list:
        print("Area of interest is a list, converting to a box")
        area_of_interest = box(*area_of_interest)
    gdf = read_file(root+'Shapefiles/water-polygons-split-4326/water_polygons.shp', bbox=area_of_interest)
    sea_polygon = gdf['geometry'].tolist()
    if simplify:
        return MultiPolygon(sea_polygon).simplify(0.01, preserve_topology=False)
    else:
        return MultiPolygon(sea_polygon).simplify(0.01, preserve_topology=True)

def bin_spatial(spatial_extent, lat_resolution=0.02, lon_resolution=0.02):
    """
    The input for bin_spatial() is given in the following order:

    bin_spatial(lat_edge_length, lat_edge_offset, lat_edge_step, lon_edge_length, lon_edge_offset, lon_edge_step)
    """
    lon_min, lat_min, lon_max, lat_max = spatial_extent
    n_lat_edge_points = int((lat_max - lat_min) / lat_resolution) + 1
    n_lon_edge_points = int((lon_max - lon_min) / lon_resolution) + 1
    return f"bin_spatial({n_lat_edge_points},{lat_min},{lat_resolution},{n_lon_edge_points},{lon_min},{lon_resolution})"

def mask_coordinates(path: str, array_to_mask: np.array, tropomi_data: harp.Product, polygon: shp.geometry.Polygon, area_of_interest: list) -> np.array:
    """
    Masks the data based on the given polygon.
    
    Args:
        path: The path to the file where the mask will be saved, ends with a '/'
        array_to_mask: The data to be masked, 
        pointsgrid: The coordinates of the data, (latitude, longitude, (longitude_coords, latitude_coords))
        polygon: Shapely polygon to be used for masking, in this case, the water in the area of interest
        area_of_interest: The area of interest in the form [lon_min, lat_min, lon_max, lat_max]

    Returns:
        The masked data
    """

    # check if mask exists as a file
    filename = path+'Mask_files/mask_' + "_".join([str(x) for x in area_of_interest]) + '.npy'
    if os.path.exists(filename):
        mask = np.load(filename)
    else:    
        # mask the data with the tile polygon
        lon_grid, lat_grid = np.meshgrid(tropomi_data.longitude.data, tropomi_data.latitude.data)
        # turn lon_grid and lat_grid into an array with shape (n*m,2)
        points = np.array([lon_grid.flatten(), lat_grid.flatten()]).T

        # Check if each point is inside the polygon, remember that the first point is the longitude and the second the latitude
        mask_flat = polygon.contains(shp.points(coords=points))
        mask = mask_flat.reshape(lon_grid.shape)
        # Save the mask to a file
        if not os.path.exists(path+'Mask_files'):
            os.makedirs(path+'Mask_files')
        np.save(filename, mask)

    # Apply the mask to the data
    return np.where(mask, array_to_mask, np.nan)

def get_regrid_operations(spatial_extent:list, cloud_fraction:float = 0.5, vmax_wind:int = 10, qa_value:int = 50,
                          resolution:float = 0.005) -> tuple:
    """
    Returns the operations and reduce_operations for the given spatial_extent.
    
    This function regrids the data to a 0.005 degree grid and filters out the data with cloud_fraction > 0.5
    and tropospheric_NO2_column_number_density_validity > 50.
    
    Args:
        spatial_extent (list): A list of 4 floats [lon_min, lat_min, lon_max, lat_max] defining the spatial extent.
        cloud_fraction (float, optional): The maximum cloud fraction allowed. Defaults to 0.5.
        vmax_wind (int, optional): The maximum wind velocity allowed. Defaults to 10.
        qa_value (int, optional): The minimum quality assurance value allowed. Defaults to 50.
        resolution (float, optional): The resolution of the grid in degrees. Defaults to 0.005.
    
    Returns:
        tuple: A tuple containing the operations (str) to be performed on the data and the reduce_operations (str) to be performed on the data.
    """
    lat_resolution = lon_resolution = resolution  # the resolution of the grid in degrees

    supp_features = [
        'surface_meridional_wind_velocity', 
        'surface_zonal_wind_velocity',
        'sensor_zenith_angle', 
        'solar_azimuth_angle', 
        'solar_zenith_angle',
    ]
    products = ['NO2', 'HCHO', 'SO2']

    keep = ",".join(supp_features+[f"{product}_slant_column_number_density" for product in products])

    operations = [
        f"cloud_fraction<{cloud_fraction}",
        f"tropospheric_NO2_column_number_density_validity>{qa_value}",
        f"tropospheric_HCHO_column_number_density_validity>{qa_value}",
        f"SO2_column_number_density_validity>{qa_value}",
        # Filter on wind, remember wind can be postive or negative
        f"surface_meridional_wind_velocity<={vmax_wind}",
        f"surface_meridional_wind_velocity>=-{vmax_wind}",
        f"surface_zonal_wind_velocity<={vmax_wind}",
        f"surface_zonal_wind_velocity>=-{vmax_wind}",
        f"keep({keep}, latitude_bounds, longitude_bounds, datetime_start, datetime_length)",
        bin_spatial(spatial_extent, lat_resolution, lon_resolution),
    ]

    operations += [f"derive({product}_slant_column_number_density [molec/cm2])" for product in products]
    operations += [f"derive({val})" for val in supp_features]
    operations += ["derive(latitude {latitude})", "derive(longitude {longitude})", "derive(datetime_start)"]

    #The actual call to regrid and merge the selected files for given area of interest.
    operations = ";".join(operations)
    reduce_operations = "squash(time, (latitude, longitude, latitude_bounds, longitude_bounds));bin()"
    return operations, reduce_operations

def create_tiles_grid(spatial_extent: list, square_size: float = 80) -> list:
    """
    Creates a grid of square tiles within the given spatial extent.

    Each tile is entirely contained within the spatial extent.

    Args:
        spatial_extent (List[float]): Bounding box as [lon_min, lat_min, lon_max, lat_max].
        square_size (float, optional): Size of each tile in kilometers. Defaults to 80.0.

    Returns:
        Tuple[
            List[Tuple[float, float, float, float]],  # List of tile bounds
            List[Polygon],                            # List of tile center polygons
            List[Polygon]                             # List of tile polygons
        ]
    """
    # Convert square_size from kilometers to degrees (approximate)
    square_size_deg = square_size / 111.32  # 1 degree ≈ 111.32 km

    # Create the main area polygon
    area_polygon = box(*spatial_extent)
    min_x, min_y, max_x, max_y = area_polygon.bounds

    # Calculate the number of tiles that fit within the spatial extent
    num_tiles_x = int((max_x - min_x) // square_size_deg)
    num_tiles_y = int((max_y - min_y) // square_size_deg)

    # Initialize lists to store tile information
    tiles = []            # List of tile bounds
    tiles_polygons = []   # List of tile polygons
    tile_centers = []     # List of center polygons of each tile

    # Generate the grid of tiles
    for i in range(num_tiles_x):
        x = min_x + i * square_size_deg
        for j in range(num_tiles_y):
            y = min_y + j * square_size_deg

            # Create the square tile polygon
            square = box(x, y, x + square_size_deg, y + square_size_deg)
            tiles_polygons.append(square)
            tiles.append(square.bounds)

            # Create the center polygon (6/8th size)
            minx_tile, miny_tile, maxx_tile, maxy_tile = square.bounds
            center_square = box(
                minx_tile + (maxx_tile - minx_tile) / 8,
                miny_tile + (maxy_tile - miny_tile) / 8,
                maxx_tile - (maxx_tile - minx_tile) / 8,
                maxy_tile - (maxy_tile - miny_tile) / 8
            )
            tile_centers.append(center_square)

    # # plot fitted squares with geopandas
    # # create geodataframe from tiles list
    # gdf = gpd.GeoDataFrame({'c':['red']*len(tiles_polygons)+['blue']}, geometry=tiles_polygons+[area_polygon])
    # gdf.boundary.plot(color=gdf['c'])
    # plt.savefig("tmp.png")
    # plt.close()

    return tiles, tile_centers, tiles_polygons

def create_tiles_random(spatial_extent: list, square_size: float = 80, n_squares: int = 50, ocean : MultiPolygon = None, plot: bool = True) -> list:
    """
    creates a list of square tiles for the given spatial_extent
    The number of squares is determined by n_squares.
    The tiles have a size of tile_size x tile_size
    The tiles are placed randomly within the bounds of the spatial_extent
    Only tiles that are completely within the ocean and the spatial_extent are kept
    
    Args:
    spatial_extent: list of 4 floats [lon_min, lat_min, lon_max, lat_max]
    tile_size: float, the size of the tiles in degrees
    n_squares: int, the number of squares to be created
    ocean: shapely.geometry.Polygon, the ocean polygon
    
    Returns:
    tiles: list of lists, each list contains the spatial_extent of a tile
    tile_centers: list of polygons, each polygon is the center area of a tile
    tiles_polygons: list of polygons, each polygon is a tile
    """
    def random_tile_in_polygon(spatial_extent, square_size):
        min_x, min_y, max_x, max_y = spatial_extent.bounds
        x = np.random.uniform(min_x, max_x-square_size)
        y = np.random.uniform(min_y, max_y-square_size)
        square = box(x, y, x+square_size, y+square_size)
        return square
    # convert square_size to degrees
    square_size = square_size / 111.32
    
    spatial_extent = box(*spatial_extent)
    tiles = []
    tiles_polygons = []
    tile_centers = []
    # Random placement of squares:
    for _ in range(n_squares):
        if ocean is None:
            square = random_tile_in_polygon(spatial_extent, square_size)
            tiles.append(square)
            
        else:
            while True:
                tile = random_tile_in_polygon(spatial_extent, square_size)
                # check if the square is fully within the ocean and the spatial_extent
                if spatial_extent.contains_properly(tile) and ocean.contains_properly(tile):
                    tiles_polygons.append(tile)
                    tile_bounds = tile.bounds
                    tiles.append(tile_bounds)
                    
                    minx_tile, miny_tile, maxx_tile, maxy_tile = tile_bounds
                    # get tile center area of size 6/8th of the tile size
                    tile_centers.append(box(
                        minx = minx_tile + (maxx_tile - minx_tile) / 8, 
                        miny = miny_tile + (maxy_tile - miny_tile) / 8, 
                        maxx = maxx_tile - (maxx_tile - minx_tile) / 8, 
                        maxy = maxy_tile - (maxy_tile - miny_tile) / 8
                    ))
                    break
            
    # if plot:
    #     # plot fitted squares with geopandas
    #     # create geodataframe from tiles list
    #     gdf = gpd.GeoDataFrame({'c':['red']*len(fitted_squares)+['blue']+['green']}, geometry=fitted_squares+[spatial_extent]+[ocean])
    #     gdf.boundary.plot(color=gdf['c'])
    #     plt.show()
            
    return tiles, tile_centers, tiles_polygons

def import_product(path: str, spatial_extent: list, resolution: float = 0.005) -> tuple:
    """
    Imports the data from the given path and spatial_extent.
    
    Args:
        path (str): The path to the data file.
        spatial_extent: A list of 4 floats [lon_min, lat_min, lon_max, lat_max] defining the spatial extent.
        resolution: The resolution of the grid in degrees. Defaults to 0.005.
    
    Returns:
        harp.Product: The data product.
        np.datetime64: The datetime of the data.
    """
    # Read the data
    time_filter = "longitude>={};latitude>={};longitude<={};latitude<={}".format(*spatial_extent)
    time_filter += ";keep(datetime_start,datetime_length)"
    try:
        time_data = harp.import_product(path, operations=time_filter)
    except harp.NoDataError:
        return None, None
    # Get the datetime of the data
    datetime = np.mean(time_data.datetime_start.data) + time_data.datetime_length.data
    # convert the datetime from seconds since 2010-01-01 to np.datetime64
    datetime = np.datetime64('2010-01-01') + np.timedelta64(int(datetime), 's')
    
    # Get the operations and reduce_operations
    operations, reduce_operations = get_regrid_operations(spatial_extent, resolution=resolution)
    try:
        data = harp.import_product(path, operations=operations, reduce_operations=reduce_operations)
    except harp.NoDataError:
        return None, None
    return data, datetime
    
def main(): 
    pass
    
if __name__ == "__main__":
    main()