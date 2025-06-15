#!/usr/bin/env python3

import yaml
import sys
import os
import math
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

def load_terrain_map(geotiff_path):
    dataset = gdal.Open(geotiff_path)
    if not dataset:
        print(f"Error: Could not open GeoTIFF file: {geotiff_path}")
        sys.exit(1)
    
    band = dataset.GetRasterBand(1)
    transform = dataset.GetGeoTransform()
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    min_x = transform[0]
    max_x = min_x + width * transform[1]
    min_y = transform[3] + height * transform[5]
    max_y = transform[3]
    
    print(f"Loaded terrain: {width}x{height}")
    print(f"Bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
    
    return dataset, band, transform, width, height

def get_terrain_height(x, y, band, transform, width, height):
    px = int((x - transform[0]) / transform[1])
    py = int((y - transform[3]) / transform[5])
    
    if px < 0 or px >= width or py < 0 or py >= height:
        return 0.0
    
    height_array = band.ReadAsArray(px, py, 1, 1)
    return float(height_array[0, 0]) if height_array is not None else 0.0

def generate_mission(uav_id, num_uavs, terrain_data, map_size, max_altitude, safe_clearance=10.0):  
    band, transform, width, height = terrain_data
    
    grid_size = int(math.ceil(math.sqrt(num_uavs)))
    row = (uav_id - 1) // grid_size
    col = (uav_id - 1) % grid_size
    
    spacing_x = map_size / (grid_size + 1)
    spacing_y = map_size / (grid_size + 1)
    
    base_x = -map_size/2 + spacing_x * (col + 1)
    base_y = -map_size/2 + spacing_y * (row + 1)
    
    mission = {
        'base': {
            'x': base_x,
            'y': base_y,
            'z': 0.0
        },
        'waypoints': []
    }
    
    cell_size_x = map_size / grid_size
    cell_size_y = map_size / grid_size
    
    waypoint_offsets = [
        (0.3, 0.3),    
        (0.3, -0.3),   
        (-0.3, -0.3),  
        (-0.3, 0.3)    
    ]
    
    waypoints = []
    
    for ox, oy in waypoint_offsets:
        wp_x = base_x + ox * cell_size_x / 2
        wp_y = base_y + oy * cell_size_y / 2
        
        wp_x = max(-map_size/2 + 20, min(map_size/2 - 20, wp_x))
        wp_y = max(-map_size/2 + 20, min(map_size/2 - 20, wp_y))
        
        terrain_height = get_terrain_height(wp_x, wp_y, band, transform, width, height)
        
        flying_altitude = terrain_height + safe_clearance
        
        if flying_altitude > max_altitude:
            flying_altitude = max_altitude
        
        waypoints.append({
            'x': float(wp_x),
            'y': float(wp_y),
            'z': float(flying_altitude)
        })
    
    mission['waypoints'] = waypoints
    return mission

def main():

    num_uavs = int(sys.argv[1])
    output_dir = sys.argv[2]
    terrain_file = sys.argv[3]
    map_size = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0
    max_altitude = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0
    
    print(f"Loading terrain from {terrain_file}...")
    dataset, band, transform, width, height = load_terrain_map(terrain_file)
    terrain_data = (band, transform, width, height)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating missions for {num_uavs} UAVs")
    print(f"Map size: {map_size}x{map_size} meters")
    print(f"Max altitude: {max_altitude} meters")
    print(f"Safe clearance: 10.0 meters above terrain")
    
    spawn_positions = []
    altitude_violations = 0
    
    for i in range(1, num_uavs + 1):
        mission = generate_mission(i, num_uavs, terrain_data, map_size, max_altitude)
        
        filename = os.path.join(output_dir, f'mission_{i}.yaml')
        with open(filename, 'w') as f:
            yaml.dump(mission, f, default_flow_style=False)
        
        spawn_positions.append((mission['base']['x'], mission['base']['y']))
        
        for wp in mission['waypoints']:
            if wp['z'] >= max_altitude:
                altitude_violations += 1
        
        if i % 10 == 0:
            print(f"Generated {i}/{num_uavs} missions...")
    
    print(f"\nGenerated {num_uavs} mission files in {output_dir}")
    
    if altitude_violations > 0:
        print(f"\nWarning: {altitude_violations} waypoints at or above max altitude ({max_altitude}m)")
    
    dataset = None

if __name__ == "__main__":
    main()