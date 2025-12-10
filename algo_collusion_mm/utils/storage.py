""" Utility module for managing data storage and loading.
"""
import copy
import json
import numpy as np
import os
import pickle

from datetime import datetime
from matplotlib.figure import Figure
from typing import Any, Dict, List



class CustomEncoder(json.JSONEncoder):
    """
    JSON encoder that adds support for NumPy arrays.
    """
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return str(obj).replace('\n', ',')
        return super().default(obj)



class ExperimentStorage:
    """
    Class for saving and loading experiment-related Python objects,
    figures and other infos into experiment-specific folders.

    Each episode is stored in a directory named:
        episode_<progressive_id>_<timestamp>

    Objects passed to the saver are serialized individually in this folder.

    Attributes
    ----------
    base_path : str or None
        Path to the root directory where experiments will be saved.
    experiment_counter : int
        Progressive ID counter for experiments within this storage instance.
    """

    def __init__(self, base_path: str|None, padding: int = 3):
        """
        Parameters
        ----------
        base_path : str or None
            Path to the root directory where episodes will be saved.
            If None, this saver can be used only to load data.
        padding : int, default=3
            Total number of digits for the episode number, padded with leading zeros.
        """
        self.base_path = base_path
        """ Path to the root directory."""
        self.padding = padding
        """ Number of digits for the episode number."""
        self.episode_counter = 0
        """ Progressive ID counter for episodes."""

        if base_path is not None:
            os.makedirs(self.base_path, exist_ok=True)
        return


    def load_objects(self, exp_dir: str) -> Dict[str, Any]:
        """
        Load all saved objects from a given folder.

        Parameters
        ----------
        exp_dir : str
            Path to the directory containing saved objects.

        Returns
        -------
        : dict of str to any
            A dictionary mapping object names (filenames without extension)
            to the loaded Python objects.
        """
        loaded = {}
        for file_name in os.listdir(exp_dir):
            if file_name.endswith('.pkl'):
                name = file_name.replace('.pkl', '')
                file_path = os.path.join(exp_dir, file_name)
                with open(file_path, 'rb') as f:
                    loaded[name] = pickle.load(f)
        return loaded


    def print_and_save(self, text: str, silent: bool = False) -> None:
        """
        Print and append a single line of text to the `RESULTS.txt` file in the base path.

        This method logs experiment-related information by printing it to the console 
        and appending it to a persistent file. Useful for tracking progress or results 
        during batch runs or long experiments.

        If the file does not exist, it is created automatically.

        Parameters
        ----------
        text : str
            The line of text to print and append to the results file.
        silent : bool, defualt=False
            If True, suppress printing to stdout.

        Raises
        ------
        ValueError
            If `base_path` is None.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        if not silent:
            print(text)
        
        file_path = os.path.join(self.base_path, 'RESULTS.txt')
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        return


    def save_episode(self, objects: List[Any]|None = None, figure: Figure|None = None, info: str|Dict|None = None) -> str:
        """
        Save a list of Python objects, a figure, and metadata 
        (string or JSON) into a newly created episode folder.

        Each object is serialized with pickle, figures are saved as PNG, 
        and info is stored as a text or JSON file.

        Parameters
        ----------
        objects : list of any or None, defualt=None
            List of Python objects to be saved. Each object may define a `name` attribute,
            which is used as filename. If not provided, the object's Python ID is used instead.
            Objects containing unserializable elements  (e.g., lambda functions) will have 
            those attributes set to `None` before pickling.
        figure : matplotlib.figure.Figure or None, defualt=None
            If given, the figure will be saved as `PLOT.png` inside the 
            experiment folder.
        info : str or dict or None, default=None
            If string, it is written to a text file named `INFO.txt`.  
            If dict, it is written to a JSON file named `INFO.json`.  
                
        Returns
        -------
        : str
            Path to the episode folder where objects were saved.
        
        Raises
        ------
        ValueError
            If `base_path` is None.
        TypeError
            If `info` is not a string or a dictionary.

        Notes
        -----
        - Each call creates a unique episode directory.
        - Objects containing non-pickleable elements (e.g., lambdas, open files,
          locally defined functions) will have those attributes replaced with `None` before serialization.
        - This method overwrites files if names already exist within the new folder.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        exp_dir = self._create_episode_dir()

        # Save objects
        if objects is not None:
            for obj in objects:
                obj_copy = copy.deepcopy(obj)

                for attr, value in obj_copy.__dict__.items():
                    if callable(value) and getattr(value, '__name__', '') == '<lambda>':
                        setattr(obj_copy, attr, None)

                file_name = getattr(obj_copy, 'name', str(id(obj_copy)))
                file_path = os.path.join(exp_dir, f'{file_name}.pkl')

                with open(file_path, 'wb') as f:
                    pickle.dump(obj_copy, f)
        
        # Save figure
        if figure is not None:
            file_path = os.path.join(exp_dir, 'PLOT.png')
            figure.savefig(file_path)

        # Save info as txt or json
        if info is not None:
            if isinstance(info, str):
                file_path = os.path.join(exp_dir, 'INFO.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(info)
            elif isinstance(info, dict):
                file_path = os.path.join(exp_dir, 'INFO.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
            else:
                raise TypeError('`info` must be either a string or a dict (JSON)')
        return exp_dir


    def save_figures(self, figures: Dict[str, Figure], dpi: int = 300) -> None:
        """
        Save a dictionary of Matplotlib figures to PNG files.

        Parameters
        ----------
        figures : dict of str to matplotlib.figure.Figure
            Dictionary where keys are string names used as filenames (without extension)
            and values are Matplotlib figure objects to be saved.
        dpi : int, default=300
            Resolution in dots per inch for the saved figures. Default is 300.

        Raises
        ------
        ValueError
            If `base_path` is None.

        Notes
        -----
        Each figure is saved as `<key>.png` in the specified `base_path`.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        for name, fig in figures.items():
            file_path = os.path.join(self.base_path, f'{name}.png')
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        return

    
    def save_objects(self, objects: Dict[str, Any]) -> None:
        """
        Save a dictionary of objects to pickle files.

        Parameters
        ----------
        objects : dict of str to any
            Dictionary where keys are string names used as filenames (without extension)
            and values are the objects to be saved.

        Raises
        ------
        ValueError
            If `base_path` is None.

        Notes
        -----
        - Each object is saved as '<key>.pkl' in the specified `base_path`.
        - Objects containing elements that cannot be pickled (e.g., lambda functions,
          open file handles, or locally defined functions) will cause a `PicklingError`.

        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        for name, obj in objects.items():
            file_path = os.path.join(self.base_path, f'{name}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        return


    def _create_episode_dir(self) -> str:
        """
        Create a new episode directory with progressive ID and timestamp.

        Returns
        -------
        : str
            Path to the created episode directory.

        Raises
        ------
        ValueError
            If `base_path` is None.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        self.episode_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(self.base_path, f'episode_{self.episode_counter:0{self.padding}}_{timestamp}')

        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
