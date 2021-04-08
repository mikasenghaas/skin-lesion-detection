import os
import pandas as pd
import json

def save_csv(data, path, filename, index=False, force=True):
    """
    Helper-Function to export pandas DataFrames into `csv` format using pandas built-in method `to_csv()`. The function provides functionality to force the creation of the path if not previously located in the file structure.

    Parameters:
        data                : pd.DataFrame
        path                : str (relative path from directory of execution file)
        filename            : str (descriptive filename (NOTE: without `.csv` file extension))
        index               : boolean (Specifies saving process in pandas.to_csv(). For more information check out the documentation of `to_csv()`)
        force               : boolean (True for automatic path creation using `os`)
    Return:
        None
    """
    if force:
        try: os.makedirs(path)
        except: None
    data.to_csv(f"{path}{filename}.csv", index=index)

def save_json(dict, path, filename, force=True):
    if force:
        try: os.makedirs(path)
        except: None

    with open(path + filename, 'w') as outfile:
        json.dump(dict, outfile)

def save_dict(dict, path, filename, force=True, save_to='csv'):
    """
    Function to save a Python dictionary into either `csv` or `json` format for further use or extensive inspection by specifying a having path and filename. 

    Parameters:
        dict            : dict (SUMMARY dict holding information about each column of the dataset)
        path            : str (Relative path to location of saving)
        filename        : str (Filename (without suffix `.csv` or `.json`))
        save_to         : str (either `csv` or `json`)
    Return: None 
    """
    dataframe = pd.DataFrame(dict)
    
    if force:
        try: os.makedirs(path)
        except: None

    if save_to == 'csv': dataframe.to_csv(f'{path}/{filename}.csv')
    elif save_to == 'json': dataframe.to_json(f'{path}/{filename}.json')
    else: raise NameError(f"'{save_to}' not defined. Try saving to 'csv' or 'json' format.")
    print(f"Saved: {filename}.{save_to} to {path}")

def save_figure(figure, path, filename, force=True, save_to='pdf'):
    """
    Function to save any matplotlib figure into a specified (relative) path and given filename. The function provides functionality to force the creation of the path if not previously located in the file structure.

    Parameters:
        figure          : plt.Figure 
        path            : str (Relative path to location of saving)
        filename        : str (Filename (without suffix `.csv` or `.json`))
        force           : boolean (Creates Path automatically if `True`, else `False`)
        save_to         : str (either `csv` or `json`)
    Return: None 
    """
    if force:
        try: os.makedirs(path)
        except: None

    figure.savefig(f'{path}/{filename}.{save_to}')
    print(f"Saved: '{filename}.{save_to}' to {path}")

# save_all_categorical_associations(data = DATA_LEEDS[dataset], severity_summary=SUMMARY['accidents'][6], dataset_name=dataset, summary=SUMMARY[dataset], severity= SEVERITY[dataset], path=PATH['reports']['leeds'] + PATH[dataset] + 'associations/')

def save_map(_map, path, filename):
    """
    Function to save a `folium.Map` object in `html` format into the specified (relative) path with the given filename.

    Parameters:
        _map        : folium.Map
        path        : str (Representing relative path to save location)
        filename    : str 
    """
    try:
        os.makedirs(path)
    except: None

    _map.save(f'{path}/{filename}.html')