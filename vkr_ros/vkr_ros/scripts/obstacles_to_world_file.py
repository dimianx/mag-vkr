#!/usr/bin/env python3

import sys
import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from osgeo import gdal
import numpy as np

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
    
    return dataset, band, transform, width, height, min_x, max_x, min_y, max_y

def get_terrain_height(x, y, band, transform, width, height):
    px = int((x - transform[0]) / transform[1])
    py = int((y - transform[3]) / transform[5])
    
    if px < 0 or px >= width or py < 0 or py >= height:
        return 0.0
    
    height_array = band.ReadAsArray(px, py, 1, 1)
    return float(height_array[0, 0]) if height_array is not None else 0.0

def parse_world_file(world_file):
    tree = ET.parse(world_file)
    root = tree.getroot()
    
    world = root.find('.//world')
    if world is None:
        print("Error: No world element found in file")
        sys.exit(1)
    
    return tree, root, world

def create_obstacle_model(obj_file, name, x, y, z, roll=0, pitch=0, yaw=0):
    model = ET.Element('model', name=name)
    model.set('name', name)
    
    static = ET.SubElement(model, 'static')
    static.text = 'true'
    
    pose = ET.SubElement(model, 'pose')
    pose.text = f'{x} {y} {z} {roll} {pitch} {yaw}'
    
    link = ET.SubElement(model, 'link', name='link')
    
    collision = ET.SubElement(link, 'collision', name='collision')
    collision_geom = ET.SubElement(collision, 'geometry')
    collision_mesh = ET.SubElement(collision_geom, 'mesh')
    collision_uri = ET.SubElement(collision_mesh, 'uri')
    collision_uri.text = f'file://{obj_file}'
    
    visual = ET.SubElement(link, 'visual', name='visual')
    visual_geom = ET.SubElement(visual, 'geometry')
    visual_mesh = ET.SubElement(visual_geom, 'mesh')
    visual_uri = ET.SubElement(visual_mesh, 'uri')
    visual_uri.text = f'file://{obj_file}'
    
    return model

def generate_random_positions(num_obstacles, map_size, min_distance=20.0, edge_margin=50.0):
    positions = []
    max_attempts = 1000
    
    for i in range(num_obstacles):
        found_valid = False
        attempts = 0
        
        while not found_valid and attempts < max_attempts:
            x = random.uniform(-map_size/2 + edge_margin, map_size/2 - edge_margin)
            y = random.uniform(-map_size/2 + edge_margin, map_size/2 - edge_margin)
            
            valid = True
            for px, py in positions:
                distance = np.sqrt((x - px)**2 + (y - py)**2)
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                positions.append((x, y))
                found_valid = True
            
            attempts += 1
        
        if not found_valid:
            print(f"Warning: Could not find valid position for obstacle {i+1}")
    
    return positions

def prettify_xml(element):
    rough_string = ET.tostring(element, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    
    world_file = sys.argv[1]
    output_file = sys.argv[2]
    terrain_file = sys.argv[3]
    obstacles_dir = sys.argv[4]
    num_obstacles = int(sys.argv[5])
    map_size = float(sys.argv[6]) if len(sys.argv) > 6 else 1000.0
    
    print(f"Loading terrain from {terrain_file}...")
    dataset, band, transform, width, height, min_x, max_x, min_y, max_y = load_terrain_map(terrain_file)
    
    print(f"Parsing world file {world_file}...")
    tree, root, world = parse_world_file(world_file)
    
    obj_files = []
    for filename in os.listdir(obstacles_dir):
        if filename.endswith('.obj'):
            obj_files.append(os.path.join(obstacles_dir, filename))
    
    if not obj_files:
        print(f"Error: No OBJ files found in {obstacles_dir}")
        sys.exit(1)
    
    print(f"Found {len(obj_files)} OBJ files")
    
    print(f"Generating {num_obstacles} random positions...")
    positions = generate_random_positions(num_obstacles, map_size)
    
    print("Adding obstacles to world...")
    obstacles_added = 0
    
    for i, (x, y) in enumerate(positions):
        z = get_terrain_height(x, y, band, transform, width, height)
        
        obj_file = random.choice(obj_files)
        obj_name = os.path.basename(obj_file).replace('.obj', '')
        
        yaw = random.uniform(0, 2 * np.pi)
        
        model_name = f"{obj_name}_{i+1}"
        
        obstacle_model = create_obstacle_model(
            obj_file, model_name, x, y, z, 0, 0, yaw
        )
        
        world.append(obstacle_model)
        obstacles_added += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Added {i + 1}/{num_obstacles} obstacles...")
    
    print(f"Writing output to {output_file}...")
    
    xml_str = '<?xml version="1.0"?>\n'
    xml_str += ET.tostring(root, encoding='unicode')
    
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    lines = pretty_xml.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    pretty_xml = '\n'.join(non_empty_lines[1:])  
    
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(pretty_xml)
    
    print(f"\nSuccessfully added {obstacles_added} obstacles to world")
    print(f"Output saved to: {output_file}")
    
    config_file = output_file.replace('.world', '_obstacles.yaml')
    print(f"\nGenerating obstacles configuration: {config_file}")
    
    with open(config_file, 'w') as f:
        f.write("# Auto-generated obstacles configuration\n")
        f.write("obstacles:\n")
        
        for i, (x, y) in enumerate(positions[:obstacles_added]):
            z = get_terrain_height(x, y, band, transform, width, height)
            obj_file = random.choice(obj_files)
            obj_name = os.path.basename(obj_file)
            yaw = random.uniform(0, 2 * np.pi)
            
            f.write(f"  - file: \"{obj_name}\"\n")
            f.write(f"    position: [{x:.2f}, {y:.2f}, {z:.2f}]\n")
            f.write(f"    rotation: [0.0, 0.0, {yaw:.4f}]\n")
            f.write(f"    scale: [1.0, 1.0, 1.0]\n\n")
    
    print(f"Configuration saved to: {config_file}")
    
    dataset = None

if __name__ == "__main__":
    main()