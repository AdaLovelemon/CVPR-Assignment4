import os
import yaml

def list_all_items(directory_path):
    items = os.listdir(directory_path)
    # join the directory path with each item
    full_paths = [os.path.join(directory_path, item) for item in items]
    return full_paths

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
