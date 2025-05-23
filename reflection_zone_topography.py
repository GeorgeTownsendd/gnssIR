import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
import contextily as ctx
from typing import Union, List, Tuple, Dict
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon as ShapelyPolygon, LineString
from rasterio.warp import transform_bounds
import rasterio.transform as rtransform
from scipy.ndimage import map_coordinates


# Previous helper functions remain unchanged
def read_phase_data(filename):
    """
    Read phase data from text file and return dictionary of satellite info.
    Format: sat_type RefH prn azimuth nval min_az max_az
    """
    phase_data = {}
    with open(filename, 'r') as f:
        # Skip header lines
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:  # Make sure we have all fields
                prn = int(parts[2])
                ref_h = float(parts[1])
                azimuth = float(parts[3])
                nval = int(parts[4])
                min_az = float(parts[5])
                max_az = float(parts[6])

                # Store all relevant azimuth info
                if prn not in phase_data:
                    phase_data[prn] = []

                phase_data[prn].append({
                    'mean_az': azimuth,
                    'nval': nval,
                    'min_az': min_az,
                    'max_az': max_az,
                    'ref_h': ref_h
                })
    return phase_data


import numpy as np
import rasterio
from typing import Tuple


def get_elevation_profile(dem_path: str, start_point: Tuple[float, float],
                          end_point: Tuple[float, float], num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract elevation profile between two points from a DEM.

    Args:
        dem_path (str): Path to the DEM GeoTIFF file
        start_point (tuple): (x, y) coordinates of start point in DEM's CRS
        end_point (tuple): (x, y) coordinates of end point in DEM's CRS
        num_points (int): Number of points to sample along the line

    Returns:
        Tuple of (distances, elevations) arrays
        - distances: Array of distances from start point (in same units as DEM CRS)
        - elevations: Array of elevation values (in same units as DEM)
    """
    with rasterio.open(dem_path) as src:
        # Create line of points between start and end
        x_coords = np.linspace(start_point[0], end_point[0], num_points)
        y_coords = np.linspace(start_point[1], end_point[1], num_points)

        # Get pixel coordinates for each point
        row_indices, col_indices = [], []
        for x, y in zip(x_coords, y_coords):
            r, c = src.index(x, y)
            row_indices.append(r)
            col_indices.append(c)

        # Convert to numpy arrays
        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)

        # Read DEM data
        dem_data = src.read(1)

        # Clip indices to valid range
        row_indices = np.clip(row_indices, 0, dem_data.shape[0] - 1)
        col_indices = np.clip(col_indices, 0, dem_data.shape[1] - 1)

        # Sample elevations directly using indices
        elevations = dem_data[row_indices, col_indices]

        # Calculate distances from start point
        distances = np.sqrt(
            (x_coords - start_point[0]) ** 2 +
            (y_coords - start_point[1]) ** 2
        )

        return distances, elevations


def load_kml_file(station_name: str) -> ET.Element:
    """
    Load and parse a KML file for a given station name.
    """
    file_path = f"data/refl_code/Files/kml/{station_name}.kml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KML file not found for station {station_name}")
    tree = ET.parse(file_path)
    return tree.getroot()


def extract_coords(placemark: ET.Element, namespace: str) -> Union[np.ndarray, Tuple[float, float]]:
    """
    Extract coordinates from a placemark element.
    Returns either polygon coordinates as numpy array or point coordinates as tuple.
    """
    # First try to find Point coordinates
    point = placemark.find(f".//{namespace}Point/{namespace}coordinates")
    if point is not None:
        coords = point.text.strip()
        lon, lat, _ = map(float, coords.split(','))
        return (lon, lat)

    # If no Point, check for Point coordinates in a different structure
    point_alt = placemark.find(f".//{namespace}coordinates")
    if point_alt is not None and ',' in point_alt.text and point_alt.text.count(',') == 2:
        coords = point_alt.text.strip()
        lon, lat, _ = map(float, coords.split(','))
        return (lon, lat)

    # Finally, try to find Polygon coordinates
    polygon = placemark.find(
        f".//{namespace}Polygon/{namespace}outerBoundaryIs/{namespace}LinearRing/{namespace}coordinates")
    if polygon is not None:
        coord_pairs = polygon.text.strip().split()
        polygon_coords = []
        for pair in coord_pairs:
            lon, lat, _ = map(float, pair.split(','))
            polygon_coords.append([lon, lat])
        return np.array(polygon_coords)

    return None


def calculate_azimuth(point1, point2):
    """
    Calculate azimuth between two points in degrees.
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    azimuth = np.degrees(np.arctan2(dx, dy))
    if azimuth < 0:
        azimuth += 360
    return azimuth


def plot_reflection_zones(station_name: str,
                          dem_path: str,
                          phase_file: str,
                          elevations: Union[int, List[int]] = None,
                          prns: Union[int, List[int]] = None,
                          az_tolerance: float = 10.0,
                          figsize_main: Tuple[int, int] = (24, 8),
                          zoom: int = 14) -> None:
    """
    Create visualization of reflection zones including:
    1. Satellite imagery with reflection zones
    2. DEM visualization with cross-section lines
    3. Elevation cross-section from nearest to furthest vertex

    Args:
        station_name (str): Station identifier
        dem_path (str): Path to the DEM GeoTIFF file
        phase_file (str): Path to the phase data file
        elevations (int or list): Elevation angle(s) to plot
        prns (int or list): PRN number(s) to plot
        az_tolerance (float): Tolerance for matching azimuths in degrees
        figsize_main (tuple): Figure size for main plot
        zoom (int): Zoom level for the basemap
    """
    # Convert single values to lists
    if isinstance(elevations, (int, float)):
        elevations = [elevations]
    if isinstance(prns, int):
        prns = [prns]

    # Load phase data
    phase_data = read_phase_data(phase_file)

    # Create colormap for Nval visualization
    nval_cmap = plt.cm.viridis

    # Find min and max Nval for color scaling
    nval_values = []
    for prn_data in phase_data.values():
        for entry in prn_data:
            nval_values.append(entry['nval'])
    min_nval = min(nval_values) if nval_values else 0
    max_nval = max(nval_values) if nval_values else 1000

    # Load KML data
    root = load_kml_file(station_name)
    namespace = root.tag.split('}')[0] + '}'
    placemarks = root.findall(f".//{namespace}Placemark")

    # Create main figure with three subplots using gridspec
    fig1 = plt.figure(figsize=figsize_main, constrained_layout=True)
    gs = fig1.add_gridspec(1, 3)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1])
    ax3 = fig1.add_subplot(gs[0, 2])

    # Lists to store transformed coordinates and data
    all_x = []
    all_y = []
    all_polygons = []
    zone_info = []

    # First get station coordinates and set up projection
    station_x = station_y = None
    for placemark in placemarks:
        name = placemark.find(f"./{namespace}name")
        if name is not None and name.text.lower() == station_name.lower():
            coords = extract_coords(placemark, namespace)
            if isinstance(coords, tuple):
                lon, lat = coords
                # Get UTM zone number
                zone_number = int((lon + 180) / 6) + 1
                # Determine hemisphere
                hemisphere = 'north' if lat >= 0 else 'south'
                # Create UTM projection string
                proj_string = f"+proj=utm +zone={zone_number} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
                # Set up transformer
                transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
                # Transform coordinates
                station_x, station_y = transformer.transform(lon, lat)
                all_x.append(station_x)
                all_y.append(station_y)
                # Plot station on both maps
                for ax in [ax1, ax2]:
                    ax.plot(station_x, station_y, marker='*', color='yellow',
                            markersize=20, markeredgecolor='red', markeredgewidth=2,
                            label='Station', zorder=100)
                break

    if station_x is None or station_y is None:
        raise ValueError("Station point not found")

    # Dictionary to store polygons by PRN/elevation for legend
    poly_types = {}

    # Process reflection zones
    for placemark in placemarks:
        name = placemark.find(f"./{namespace}name")
        if name is None or name.text.lower() == station_name.lower():
            continue

        # Parse PRN and elevation
        try:
            prn_str, elev_str = name.text.split()
            current_prn = int(prn_str.split(':')[1])
            current_elev = int(elev_str.split(':')[1])
        except (ValueError, IndexError):
            continue

        # Apply filters
        if (elevations and current_elev not in elevations) or \
                (prns and current_prn not in prns):
            continue

        coords = extract_coords(placemark, namespace)
        if coords is not None and isinstance(coords, np.ndarray):
            # Transform coordinates to UTM
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            transformed_coords = np.column_stack((x, y))

            # Store coordinates for extent calculation
            all_x.extend(x)
            all_y.extend(y)

            # Calculate center point and azimuth
            center_x, center_y = np.mean(transformed_coords, axis=0)
            azimuth = calculate_azimuth((station_x, station_y), (center_x, center_y))

            # Check for matching phase data
            nval = None
            matched = False
            if current_prn in phase_data:
                for phase_entry in phase_data[current_prn]:
                    mean_az = phase_entry['mean_az']
                    min_az = phase_entry['min_az']
                    max_az = phase_entry['max_az']

                    in_range = False
                    if min_az <= max_az:
                        in_range = min_az <= azimuth <= max_az
                    else:  # Handles case where range crosses 360°
                        in_range = azimuth >= min_az or azimuth <= max_az

                    if in_range or abs(azimuth - mean_az) <= az_tolerance:
                        nval = phase_entry['nval']
                        ref_h = phase_entry['ref_h']
                        matched = True
                        break

            # Set polygon color and alpha based on matching and Nval
            if matched:
                color = nval_cmap((nval - min_nval) / (max_nval - min_nval))
                alpha = 0.6
                edge_color = 'yellow'
            else:
                color = 'gray'
                alpha = 0.3
                edge_color = 'red'

            # Create polygons for both plots
            polygon1 = Polygon(transformed_coords,
                               facecolor=color,
                               alpha=alpha,
                               edgecolor=edge_color,
                               linewidth=1,
                               label=f'PRN {current_prn}, Elev {current_elev}°' \
                                   if (current_prn, current_elev) not in poly_types else None)

            polygon2 = Polygon(transformed_coords,
                               facecolor='gray',
                               alpha=0.3,
                               edgecolor='red',
                               linewidth=1)

            # Add text label at center
            label_text = f'PRN {current_prn}\nElev {current_elev}°\nAz {azimuth:.1f}°'
            if matched:
                label_text += f'\nNval {nval}\nRefH {ref_h:.2f}m'
            else:
                label_text += '\n(No match)'

            ax1.text(center_x, center_y, label_text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='white',
                     fontsize=8,
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1),
                     zorder=99)

            poly_types[(current_prn, current_elev)] = polygon1

            # Add to plots
            ax1.add_patch(polygon1)
            ax2.add_patch(polygon2)

            # Store polygon for analysis
            all_polygons.append({
                'coords': transformed_coords,
                'shapely': ShapelyPolygon(transformed_coords),
                'info': (current_prn, current_elev, azimuth, nval, matched)
            })

    if not poly_types:
        raise ValueError("No polygons found matching the specified criteria")

    # Add colorbar for Nval
    sm = plt.cm.ScalarMappable(cmap=nval_cmap, norm=plt.Normalize(min_nval, max_nval))
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Number of Reflections (Nval)')

    # Calculate and set common extent
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    padding_x = (max_x - min_x) * 0.05
    padding_y = (max_y - min_y) * 0.05

    # Set extent for both map plots
    ax1.set_xlim(min_x - padding_x, max_x + padding_x)
    ax1.set_ylim(min_y - padding_y, max_y + padding_y)
    ax2.set_xlim(min_x - padding_x, max_x + padding_x)
    ax2.set_ylim(min_y - padding_y, max_y + padding_y)

    # Add basemap to first subplot (contextily uses Web Mercator)
    ctx.add_basemap(ax1, crs=proj_string, source=ctx.providers.Esri.WorldImagery, zoom=zoom)

    # Plot DEM in second subplot
    with rasterio.open(dem_path) as src:
        # Get the window bounds in the DEM's CRS
        dem_transformer = Transformer.from_crs(proj_string, src.crs, always_xy=True)

        # Transform the UTM extent corners to DEM CRS
        left, bottom = dem_transformer.transform(min_x - padding_x, min_y - padding_y)
        right, top = dem_transformer.transform(max_x + padding_x, max_y + padding_y)

        # Get pixel coordinates
        window = src.window(left, bottom, right, top)
        window = window.round_lengths()

        # Read the data in the window
        dem_data = src.read(1, window=window)

        # Handle nodata values
        if src.nodata is not None:
            dem_data = np.ma.masked_equal(dem_data, src.nodata)

        # Get the transform for the windowed data
        window_transform = src.window_transform(window)

        # Calculate the extent in UTM coordinates for imshow
        window_bounds = rasterio.windows.bounds(window, src.transform)
        utm_transformer = Transformer.from_crs(src.crs, proj_string, always_xy=True)

        # Transform all four corners - window_bounds returns (left, bottom, right, top)
        corners = [(window_bounds[0], window_bounds[1]),  # left, bottom
                   (window_bounds[0], window_bounds[3]),  # left, top
                   (window_bounds[2], window_bounds[3]),  # right, top
                   (window_bounds[2], window_bounds[1])]  # right, bottom

        utm_corners_x = []
        utm_corners_y = []
        for x, y in corners:
            utm_x, utm_y = utm_transformer.transform(x, y)
            utm_corners_x.append(utm_x)
            utm_corners_y.append(utm_y)

        extent = [min(utm_corners_x), max(utm_corners_x),
                  min(utm_corners_y), max(utm_corners_y)]

        # Plot the DEM
        vmin, vmax = np.nanmin(dem_data), np.nanmax(dem_data)
        im = ax2.imshow(dem_data,
                        extent=extent,
                        cmap='terrain',
                        interpolation='nearest',
                        vmin=vmin,
                        vmax=vmax)
        plt.colorbar(im, ax=ax2, label='Elevation (m)')

    # Create elevation profiles and plot cross-section lines
    ax3.clear()

    for i, polygon_data in enumerate(all_polygons):
        polygon_coords = polygon_data['coords']
        prn, elev, azimuth, nval, matched = polygon_data['info']

        # Find closest and furthest vertices from station
        distances_to_station = np.sqrt(
            (polygon_coords[:, 0] - station_x) ** 2 +
            (polygon_coords[:, 1] - station_y) ** 2
        )
        closest_vertex = polygon_coords[np.argmin(distances_to_station)]
        furthest_vertex = polygon_coords[np.argmax(distances_to_station)]

        # Get elevation profile
        with rasterio.open(dem_path) as src:
            # Convert profile endpoints from UTM to the DEM's CRS
            dem_transformer = Transformer.from_crs(proj_string, src.crs, always_xy=True)
            start_x, start_y = dem_transformer.transform(closest_vertex[0], closest_vertex[1])
            end_x, end_y = dem_transformer.transform(furthest_vertex[0], furthest_vertex[1])

            # Create the line with enough points for smooth sampling
            line_length = np.sqrt((furthest_vertex[0] - closest_vertex[0]) ** 2 +
                                  (furthest_vertex[1] - closest_vertex[1]) ** 2)
            num_points = int(line_length / 10)  # Sample every 10 meters
            num_points = max(100, num_points)  # Ensure minimum number of points

            line = LineString([(start_x, start_y), (end_x, end_y)])
            distances = np.linspace(0, line_length, num_points)
            points = [line.interpolate(d / line_length, normalized=True) for d in distances]

            # Sample elevations at these points
            elevations = []
            for point in points:
                row, col = src.index(point.x, point.y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    elevation = src.read(1, window=((row, row + 1), (col, col + 1)))[0][0]
                    if elevation != src.nodata:
                        elevations.append(elevation)
                    else:
                        elevations.append(np.nan)
                else:
                    elevations.append(np.nan)

            elevations = np.array(elevations)

        # Plot elevation profile
        color = plt.cm.tab10(i % 10)
        label = f'PRN {prn}, Elev {elev}°'
        if matched:
            label += f' (Nval: {nval})'

        # Plot cross-section line on DEM plot (no legend)
        ax2.plot([closest_vertex[0], furthest_vertex[0]],
                 [closest_vertex[1], furthest_vertex[1]],
                 '--', color=color, linewidth=2)

        # Plot elevation profile with varying opacity based on match status
        alpha = 1.0 if matched else 0.3
        label = f'PRN {prn}, Elev {elev}°, Az {azimuth:.1f}°'
        if matched:
            label += f' (Nval: {nval})'
        ax3.plot(distances, elevations, '-', color=color, label=label,
                 linewidth=2, alpha=alpha)

    # Customize plots
    ax1.set_title(f'Reflection Zones (Satellite)')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.grid(False)
    ax1.ticklabel_format(style='plain')

    ax2.set_title('Digital Surface Model')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.grid(False)
    ax2.ticklabel_format(style='plain')

    ax3.set_title('Elevation Cross-Section per Refl-Zone')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Elevation (m)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize='small')

    plt.show()
    #plt.savefig(f'/home/george/Documents/Work/{station_name}.png')
    #return fig1, (ax1, ax2, ax3)


if __name__ == "__main__":
    for station_name in ['wark']:#['ktia', 'wark', 'sedd', 'mchl']:
        #station_name = 'sedd'
        phase_file = f"data/refl_code/input/{station_name}_phaseRH.txt"
        dem_paths = {
            'ktia': "/home/george/Downloads/lds-northland-lidar-1m-dsm-2018-2020-GTiff/DSM_AV26_2018_1000_1027.tif",
            'wark': "/home/george/Downloads/lds-auckland-lidar-1m-dsm-2013-GTiff/DSM_AZ31_2236_2013.tif",
            'sedd': '/home/george/Downloads/lds-marlborough-lidar-1m-dsm-2018-GTiff(1)/DSM_BR29_2018_1000_2723.tif',
            'mchl': '/home/george/Downloads/1_Second_DSM_135682/1_Second_DSM.tif'
        }

        dem_path = dem_paths[station_name]

        plot_reflection_zones(station_name, dem_path, phase_file, elevations=[5], prns=[8], zoom=18, az_tolerance=10)