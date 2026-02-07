"""
Configuration loader
"""


# =============================================================================
# %% DESCRITPION
# =============================================================================
"""
module create static and config
"""

# =============================================================================
# %% IMPORTS
# =============================================================================

import os
import json
import yaml
import pandas as pd

# =============================================================================
# %% CONFIG
# =============================================================================

def get_config(path='config/config.yaml',env=None):
    
    # Determine the file format
    file_format = path.split('.')[-1].lower()

    # Load config based on file format
    with open(path, 'r') as f:
        if file_format == 'json':
            config = json.load(f)
        elif file_format == 'yaml' or file_format == 'yml':
            config = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Please use JSON or YAML.")

    # Set database config
    if 'database' in config.keys():
        db_config_path = config['database'].get('config_path',config['database'].get('alternative_config_path',None))
        if db_config_path:
            try:
                with open(db_config_path) as json_file:
                    config['database']['config'] = json.load(json_file)
            except:
                config['database']['config'] = None


    # Modify sender email
    if env is not None:
        try:
            email_sender = config['email']['sender']
            config['email']['sender'] = f'{env}_{email_sender}'
        except KeyError:
            pass

    # Set datetime now stamp
    config['datetime_now_str'] = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
 
    return config   