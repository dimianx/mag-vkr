#!/usr/bin/env python3

import yaml
import sys
import os

def load_obstacles_config(config_file, obj_dir):
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    obstacle_files = []
    obstacle_transforms = []
    
    for obstacle in config['obstacles']:
        obj_file = os.path.join(obj_dir, obstacle['file'])
        
        if not os.path.exists(obj_file):
            print(f"Warning: {obj_file} not found, skipping...")
            continue
            
        obstacle_files.append(obj_file)
        
        transform = []
        transform.extend(obstacle['position'])
        transform.extend(obstacle['rotation'])
        transform.extend(obstacle['scale'])
        
        obstacle_transforms.extend(transform)
    
    files_str = str(obstacle_files)
    transforms_str = str(obstacle_transforms)
    
    return files_str, transforms_str

def main():
    
    config_file = sys.argv[1]
    obj_dir = sys.argv[2]
    
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found")
        sys.exit(1)
    
    if not os.path.exists(obj_dir):
        print(f"Error: OBJ directory {obj_dir} not found")
        sys.exit(1)
    
    files_str, transforms_str = load_obstacles_config(config_file, obj_dir)
    
    print("Obstacle files:")
    print(files_str)
    print("\nObstacle transforms:")
    print(transforms_str)

if __name__ == "__main__":
    main()