import numpy as np
import matplotlib.pyplot as plt
import rasterio
import contextily as ctx
from pyproj import Transformer
from matplotlib.gridspec import GridSpec
from rasterio.transform import rowcol
import simplekml
import os


def get_elevation_profile(src, start_point, end_point, num_points=100):
    """
    Extract elevation profile between two points from a DEM.
    """
    # Create line of points
    x_coords = np.linspace(start_point[0], end_point[0], num_points)
    y_coords = np.linspace(start_point[1], end_point[1], num_points)

    # Get pixel coordinates using rowcol instead of index
    rows, cols = rowcol(src.transform, x_coords, y_coords)

    # Convert to arrays
    rows = np.array(rows)
    cols = np.array(cols)

    # Get valid indices
    valid = (
            (rows >= 0) & (rows < src.height) &
            (cols >= 0) & (cols < src.width)
    )

    # Read elevations
    elevations = np.full(num_points, np.nan)
    if np.any(valid):
        # Read the full data once
        valid_data = src.read(1)
        # Sample the points
        elevations[valid] = valid_data[rows[valid], cols[valid]]

    # Calculate distances
    distances = np.sqrt(
        (x_coords - start_point[0]) ** 2 +
        (y_coords - start_point[1]) ** 2
    )

    return distances, elevations


def analyze_terrain(dem_path: str, lat: float, lon: float, distance_m: float = 60,
                    output_dir: str = '.'):
    """
    Analyze terrain and save outputs as PNG and KML.

    Args:
        dem_path (str): Path to the DEM GeoTIFF file
        lat (float): Latitude of the point of interest
        lon (float): Longitude of the point of interest
        distance_m (float): Distance in meters to analyze in each direction (default: 60m)
        output_dir (str): Directory to save outputs (default: current directory)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create KML object
    kml = simplekml.Kml()

    with rasterio.open(dem_path) as src:
        # Set up transformers
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = 'north' if lat >= 0 else 'south'
        proj_string = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

        # Transform coordinates to DEM's CRS
        to_dem = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        center_x, center_y = to_dem.transform(lon, lat)

        # Create transformer from DEM CRS back to WGS84
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        # Calculate end points (45° angles from N)
        angles = np.array([45, 135, 225, 315])  # NE, SE, SW, NW
        rad_angles = np.radians(angles)
        dx = distance_m * np.sin(rad_angles)  # sin for E/W component
        dy = distance_m * np.cos(rad_angles)  # cos for N/S component

        # Get profiles
        profiles = {}
        directions = ['NE', 'SE', 'SW', 'NW']
        colors = ['red', 'green', 'blue', 'purple']

        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 2)

        # Map view subplot
        ax_map = fig.add_subplot(gs[0, 0])

        # Get window for display
        row, col = rowcol(src.transform, center_x, center_y)
        window_size = int(distance_m / src.res[0]) * 2
        window = rasterio.windows.Window(
            int(col - window_size // 2),
            int(row - window_size // 2),
            window_size,
            window_size
        )

        # Read and display DEM
        dem_data = src.read(1, window=window)
        window_transform = src.window_transform(window)
        bounds = rasterio.windows.bounds(window, src.transform)

        im = ax_map.imshow(dem_data,
                           extent=[bounds[0], bounds[2],  # left, right
                                   bounds[1], bounds[3]],  # bottom, top
                           cmap='viridis',
                           vmin=49,
                           vmax=53)
        plt.colorbar(im, ax=ax_map, label='Elevation (m)')

        # Plot center point
        ax_map.plot(center_x, center_y, 'k*', markersize=15, label='Center Point')

        # Add center point to KML
        center_lon, center_lat = to_wgs84.transform(center_x, center_y)
        pnt = kml.newpoint(name="Center Point")
        pnt.coords = [(center_lon, center_lat)]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png'

        # Elevation profiles subplot
        ax_profiles = fig.add_subplot(gs[0, 1])

        # Get and plot profiles
        center_elev = None
        for direction, dx_i, dy_i, color in zip(directions, dx, dy, colors):
            end_point = (center_x + dx_i, center_y + dy_i)
            distances, elevations = get_elevation_profile(
                src,
                (center_x, center_y),
                end_point,
                num_points=200
            )

            # Store profile data
            profiles[direction] = {
                'distances': distances,
                'elevations': elevations,
                'end_point': end_point
            }

            # Plot profile line on map
            end_x, end_y = end_point
            ax_map.plot([center_x, end_x], [center_y, end_y],
                        color=color, linestyle='--', label=direction)

            # Add line to KML
            end_lon, end_lat = to_wgs84.transform(end_x, end_y)
            line = kml.newlinestring(name=f"Profile {direction}")
            line.coords = [(center_lon, center_lat), (end_lon, end_lat)]
            line.style.linestyle.color = simplekml.Color.hex(color)  # Convert matplotlib color to KML
            line.style.linestyle.width = 3

            # Plot elevation profile
            ax_profiles.plot(distances, elevations,
                             color=color, linewidth=2, label=direction)

            # Store center elevation for reference
            if center_elev is None and not np.isnan(elevations[0]):
                center_elev = elevations[0]

        # Customize map subplot
        ax_map.legend()
        ax_map.set_title(f'Terrain Map with {distance_m}m Cross-Section Lines')
        ax_map.set_xlabel('Easting (m)')
        ax_map.set_ylabel('Northing (m)')

        # Customize profiles subplot
        ax_profiles.grid(True, alpha=0.3)
        ax_profiles.legend()
        ax_profiles.set_title('Elevation Profiles')
        ax_profiles.set_xlabel('Distance from Center Point (m)')
        ax_profiles.set_ylabel('Elevation (m)')

        if center_elev is not None:
            ax_profiles.axhline(y=center_elev, color='k', linestyle=':', alpha=0.5,
                                label='Center Point Elevation')
            ax_profiles.legend()

        plt.tight_layout()

        # Save outputs
        base_name = os.path.splitext(os.path.basename(dem_path))[0]
        png_path = os.path.join(output_dir, f'terrain_analysis_{base_name}.png')
        kml_path = os.path.join(output_dir, f'profile_lines_{base_name}.kml')

        # Save figure
        plt.savefig(png_path, dpi=300, bbox_inches='tight')

        # Save KML
        kml.save(kml_path)

        print(f"Saved PNG to: {png_path}")
        print(f"Saved KML to: {kml_path}")

        return fig, profiles


if __name__ == "__main__":
    # Example usage
    dem_path = "/home/george/Documents/Work/warkwork_assessment/warkwork_dsm.tif"
    lat = -36.4304816524546 #-36.7009301442°, 174.5559400139°
    lon = 174.6666123167646
    output_dir = "/home/george/Downloads"  # Specify your output directory
    fig, profiles = analyze_terrain(dem_path, lat, lon, output_dir=output_dir)  # Uses default 60m
    plt.show()