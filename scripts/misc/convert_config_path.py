import os
import yaml
def replace_linux_paths_with_windows(config_data):
    for key, value in config_data.items():
        if isinstance(value, dict):
            replace_linux_paths_with_windows(value)
        elif isinstance(value, str) and value.startswith('/'):
            if value.startswith('/mnt/wunas/'):
                config_data[key] = value.replace('/mnt/wunas/', 'Z:\\')
            elif value.startswith('/mnt/wunas2/'):
                config_data[key] = value.replace('/mnt/wunas2/', 'Y:\\')
            elif value.startswith('/home/grover/'):
                config_data[key] = value.replace('/home/grover/', 'C:\\Users\\Grover\\')
            config_data[key] = config_data[key].replace('/', '\\')
def convert_linux_to_windows_paths(config_file):
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    replace_linux_paths_with_windows(config_data)
    with open(config_file, 'w') as file:
        yaml.safe_dump(config_data, file)
# Usage
config_file = r'Z:\mfk\basty-projects\main_cfg.yaml'  # Replace with your YAML config file
convert_linux_to_windows_paths(config_file)









