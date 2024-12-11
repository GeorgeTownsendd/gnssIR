import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pyproj import Transformer
import contextily as ctx
from typing import Union, List, Tuple, Dict
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon as ShapelyPolygon


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


def plot_reflections_with_elevation(station_name: str,
                                    dem_path: str,
                                    elevations: Union[int, List[int]] = None,
                                    prns: Union[int, List[int]] = None,
                                    figsize: Tuple[int, int] = (24, 8),
                                    zoom: int = 14) -> None:
    """
    Plot reflection zones, DEM, and elevation histogram using NZTM2000 projection.

    Args:
        station_name (str): Station identifier
        dem_path (str): Path to the DEM GeoTIFF file
        elevations (int or list): Elevation angle(s) to plot
        prns (int or list): PRN number(s) to plot
        figsize (tuple): Figure size in inches
        zoom (int): Zoom level for the basemap (higher number = more detail)
    """
    # Convert single values to lists
    if isinstance(elevations, (int, float)):
        elevations = [elevations]
    if isinstance(prns, int):
        prns = [prns]

    # Set up coordinate transformer (WGS84 to NZTM2000)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

    # Load KML data
    root = load_kml_file(station_name)
    namespace = root.tag.split('}')[0] + '}'
    placemarks = root.findall(f".//{namespace}Placemark")

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Lists to store transformed coordinates for determining plot extent
    all_x = []
    all_y = []
    all_polygons = []  # Store polygons for DEM analysis

    # First find and plot the station point
    station_x = station_y = None
    for placemark in placemarks:
        name = placemark.find(f"./{namespace}name")
        if name is not None and name.text.lower() == station_name.lower():
            coords = extract_coords(placemark, namespace)
            if isinstance(coords, tuple):
                station_x, station_y = transformer.transform(coords[0], coords[1])
                all_x.append(station_x)
                all_y.append(station_y)
                ax1.plot(station_x, station_y, marker='*', color='yellow',
                         markersize=20, markeredgecolor='red', markeredgewidth=2,
                         label='Station', zorder=100)
                ax2.plot(station_x, station_y, marker='*', color='yellow',
                         markersize=20, markeredgecolor='red', markeredgewidth=2,
                         label='Station', zorder=100)
            break

    # Dictionary to store polygons by PRN/elevation for legend
    poly_types = {}

    # Now process all reflection zones
    for placemark in placemarks:
        name = placemark.find(f"./{namespace}name")
        if name is None or name.text.lower() == station_name.lower():
            continue

        # Parse PRN and elevation for polygons
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
            # Transform coordinates to NZTM2000
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            transformed_coords = np.column_stack((x, y))

            # Store coordinates for extent calculation
            all_x.extend(x)
            all_y.extend(y)

            # Create polygon for satellite map
            polygon1 = Polygon(transformed_coords,
                               alpha=0.4,
                               edgecolor='yellow',
                               linewidth=1,
                               label=f'PRN {current_prn}, Elev {current_elev}°' \
                                   if (current_prn, current_elev) not in poly_types else None)

            # Create polygon for DEM plot
            polygon2 = Polygon(transformed_coords,
                               alpha=0.4,
                               edgecolor='red',
                               linewidth=1,
                               label=f'PRN {current_prn}, Elev {current_elev}°' \
                                   if (current_prn, current_elev) not in poly_types else None)

            # Store polygon type for legend
            poly_types[(current_prn, current_elev)] = polygon1

            # Add to plots
            ax1.add_patch(polygon1)
            ax2.add_patch(polygon2)

            # Store polygon for DEM analysis
            all_polygons.append(ShapelyPolygon(transformed_coords))

    if not poly_types:
        raise ValueError("No polygons found matching the specified criteria")

    # Calculate the extent of all polygons
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add some padding (5%)
    padding_x = (max_x - min_x) * 0.05
    padding_y = (max_y - min_y) * 0.05

    # Set axis limits for both map plots
    ax1.set_xlim(min_x - padding_x, max_x + padding_x)
    ax1.set_ylim(min_y - padding_y, max_y + padding_y)
    ax2.set_xlim(min_x - padding_x, max_x + padding_x)
    ax2.set_ylim(min_y - padding_y, max_y + padding_y)

    # Add the satellite imagery basemap to first subplot
    ctx.add_basemap(ax1, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=zoom)

    # Process DEM data for elevation visualization and histogram
    with rasterio.open(dem_path) as src:
        # Read the full DEM for the area
        window = src.window(*src.bounds)
        dem_data = src.read(1, window=window)
        dem_bounds = src.bounds

        # Plot raw DEM in middle subplot
        im = ax2.imshow(dem_data,
                        extent=[dem_bounds.left, dem_bounds.right,
                                dem_bounds.bottom, dem_bounds.top],
                        cmap='terrain',
                        interpolation='nearest',
                        vmin=75,
                        vmax=90)
        plt.colorbar(im, ax=ax2, label='Elevation (m)')

        # Create a list of GeoJSON-like geometry dictionaries for histogram
        geoms = [{"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}
                 for poly in all_polygons]

        # Mask the raster with polygons for histogram
        out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata)
        valid_elevations = out_image[out_image != src.nodata]

        # Create histogram in third subplot
        ax3.hist(valid_elevations.flatten(), bins=50, edgecolor='black')
        ax3.set_title('Elevation Distribution\nin Reflection Zones')
        ax3.set_xlabel('Elevation (m)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

    # Customize first subplot (Satellite imagery)
    ax1.set_title(f'Reflection Zones\n(Satellite)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.grid(False)
    ax1.ticklabel_format(style='plain')

    # Customize second subplot (DEM)
    ax2.set_title('Reflection Zones\n(Digital Elevation Model)')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.grid(False)
    ax2.ticklabel_format(style='plain')

    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2, ax3)


def calculate_azimuth(point1, point2):
    """
    Calculate azimuth between two points in degrees.
    point1: tuple of (x, y) for starting point
    point2: tuple of (x, y) for ending point
    Returns: azimuth in degrees (0-360)
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    azimuth = np.degrees(np.arctan2(dx, dy))
    if azimuth < 0:
        azimuth += 360
    return azimuth


def plot_reflection_zone_histograms(station_name: str,
                                    dem_path: str,
                                    elevations: Union[int, List[int]] = None,
                                    prns: Union[int, List[int]] = None,
                                    figsize: Tuple[int, int] = (12, 12)) -> None:
    """
    Create a 2x2 figure showing histograms for each reflection zone.
    """
    # Set up coordinate transformer (WGS84 to NZTM2000)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

    # Load KML data
    root = load_kml_file(station_name)
    namespace = root.tag.split('}')[0] + '}'
    placemarks = root.findall(f".//{namespace}Placemark")

    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.ravel()

    # First get station coordinates
    station_coords = None
    for placemark in placemarks:
        name = placemark.find(f"./{namespace}name")
        if name is not None and name.text.lower() == station_name.lower():
            coords = extract_coords(placemark, namespace)
            if isinstance(coords, tuple):
                station_coords = transformer.transform(coords[0], coords[1])
                break

    if station_coords is None:
        raise ValueError("Station point not found")

    # Collect all elevation values and calculate azimuths
    all_elevations = []
    zone_info = []  # Store PRN, elevation angles, and azimuths

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
            # Transform coordinates to NZTM2000
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            transformed_coords = np.column_stack((x, y))

            # Calculate center point of polygon
            center_point = (np.mean(x), np.mean(y))

            # Calculate azimuth from station to center of reflection zone
            azimuth = calculate_azimuth(station_coords, center_point)

            # Create polygon for masking
            polygon = ShapelyPolygon(transformed_coords)
            geom = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}

            # Extract elevation data for this zone
            with rasterio.open(dem_path) as src:
                out_image, out_transform = mask(src, [geom], crop=True, nodata=src.nodata)
                valid_elevations = out_image[out_image != src.nodata]
                all_elevations.append(valid_elevations.flatten())
                zone_info.append((current_prn, current_elev, azimuth))

    if not all_elevations:
        raise ValueError("No valid reflection zones found")

    # Calculate common bin edges using all elevation data
    min_elev = min(np.min(elev) for elev in all_elevations)
    max_elev = max(np.max(elev) for elev in all_elevations)
    bin_edges = np.linspace(min_elev, max_elev, 51)  # 50 bins

    # Find the maximum count using the common bin edges
    max_count = 0
    for elevations in all_elevations:
        counts, _ = np.histogram(elevations, bins=bin_edges)
        max_count = max(max_count, np.max(counts))

    # Create histograms with common bins and limits
    n_zones = len(all_elevations)
    for i, (elevations, (prn, elev, azimuth)) in enumerate(zip(all_elevations, zone_info)):
        if i >= 4:  # Only show first 4 zones
            break

        # Create histogram with common bin edges
        counts, _, _ = axs[i].hist(elevations, bins=bin_edges, edgecolor='black')
        axs[i].set_title(f'PRN {prn}, Elevation {elev}°\nAzimuth: {azimuth:.1f}°')
        axs[i].set_xlabel('Elevation (m)')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(True, alpha=0.3)

        # Set common axis limits
        axs[i].set_xlim(min_elev, max_elev)
        axs[i].set_ylim(0, max_count * 1.1)  # Add 10% padding to y-axis

    # Fill empty subplots if less than 4 zones
    for i in range(min(n_zones, 4), 4):
        axs[i].set_visible(False)

    plt.tight_layout()
    return fig, axs



if __name__ == "__main__":
    # Example usage
    dem_path = "/home/george/Downloads/lds-northland-lidar-1m-dsm-2018-2020-GTiff/DSM_AV26_2018_1000_1027.tif"
    fig, (ax1, ax2, ax3) = plot_reflections_with_elevation('ktia', dem_path, elevations=[5], prns=[1], zoom=18)
    plt.show()

    # Detailed histogram analysis
    fig2, axs = plot_reflection_zone_histograms('ktia', dem_path, elevations=[5], prns=[1])
    plt.show()