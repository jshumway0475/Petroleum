import yaml
import os

def load_config(file_path):
    """Load and return the configuration from a YAML file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration data as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Default configuration path
DEFAULT_CONFIG_PATH = '/app/conduit/config/analytics_config.yaml'

def get_config(section=None, name=None, path=DEFAULT_CONFIG_PATH):
    """
    Get configuration by loading the YAML file from a given path. Optionally return a specific section.
    If the section is 'credentials', a 'name' must be specified.

    Args:
        path (str): The path to the configuration file.
        section (str, optional): Specific section of the configuration to return.
        name (str, optional): Name of the specific credential to return if section is 'credentials'.

    Returns:
        dict: The requested part of the configuration data, or the entire configuration if no section is specified.
        Raises an exception if the section is 'credentials' and no name is provided, or if the name does not exist.
    """
    config = load_config(path)
    if section:
        try:
            section_data = config[section]
        except KeyError:
            raise KeyError(f"Section '{section}' not found in the configuration.")
        
        if section == 'credentials':
            if not name:
                raise ValueError("Name must be specified for credential access.")
            try:
                return next(cred for cred in section_data if cred['name'] == name)
            except StopIteration:
                raise KeyError(f"Credential with name '{name}' not found.")

        return section_data
    
    return config
