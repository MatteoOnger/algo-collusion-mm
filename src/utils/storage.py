import copy
import json
import os
import pickle

from datetime import datetime
from matplotlib.figure import Figure
from typing import Any, List, Dict



class ExperimentStorage:
    """
    Class for saving and loading Python objects into experiment-specific folders.

    Each experiment is stored in a directory named:
        experiment_<progressive_id>_<timestamp>

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
        Initialize the storage with a base directory.

        Parameters
        ----------
        base_path : str or None
            Path to the root directory where experiments will be saved.
            If None, this saver can be used only to load data.
        padding : int, default=3
            Total number of digits for the experiment number, padded with leading zeros.
        """
        self.base_path = base_path
        """ Path to the root directory.
        """
        self.padding = padding
        """ Number of digits for the experiment number.
        """
        self.experiment_counter = 0
        """ Progressive ID counter for experiments.
        """

        if base_path is not None:
            os.makedirs(self.base_path, exist_ok=True)
        return


    def load_objects(self, exp_dir: str) -> Dict[str, Any]:
        """
        Loads all saved objects from a given experiment folder.

        Parameters
        ----------
        exp_dir : str
            Path to the experiment directory containing saved objects.

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
        Print and append a single line of text to the RESULTS.txt file in the base path.

        This method logs experiment-related information by printing it to the console 
        and appending it to a persistent file. Useful for tracking progress or results 
        during batch runs or long experiments.

        If the file does not exist, it is created automatically.

        Parameters
        ----------
        text : str
            The line of text to be printed and appended to the results file.
        silent : bool
            If True, don't print to stdout, just save.

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


    def save_experiment(self, objects: List[Any]|None = None, figure: Figure|None = None, info: str|dict|None = None) -> str:
        """
        Save a list of Python objects, an optional figure, and optional metadata 
        (string or JSON) into a newly created experiment folder.

        Each object is serialized with pickle, figures are saved as PNG, 
        and info is stored as a text or JSON file.

        Parameters
        ----------
        objects : list of any or None, defualt=None
            List of Python objects to be saved. Each object may define a 
            `name` attribute, which is used as filename. If not provided, 
            the object's Python ID is used instead.
        figure : matplotlib.figure.Figure or None, defualt=None
            If given, the figure will be saved as `PLOT.png` inside the 
            experiment folder.
        info : str or dict or None, default=None
            If string, it is written to a text file named `INFO.txt`.  
            If dict, it is written to a JSON file named `INFO.json`.  
                
        Returns
        -------
        : str
            Path to the experiment folder where objects were saved.
        
        Raises
        ------
        TypeError
            If `info` is not a string or a dictionary.
        ValueError
            If `base_path` is None.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        exp_dir = self._create_experiment_dir()

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
                    json.dump(info, f, indent=4, ensure_ascii=False)
            else:
                raise TypeError('`info` must be either a string or a dict (JSON)')
        return exp_dir


    def save_figures(self, figures: Dict[str, Any], dpi: int = 300) -> None:
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
        Each figure is saved in a separate PNG file named '<key>.png' inside `base_path`.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        for name, fig in figures.items():
            file_path = os.path.join(self.base_path, f'{name}.png')
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        return

    
    def save_objects(self, objects: Dict[str, Any]) -> None:
        """
        Save a dictionary of Python objects to pickle files.

        Parameters
        ----------
        objects : dict of str to any
            Dictionary where keys are string names used as filenames (without extension)
            and values are the Python objects to be saved.

        Raises
        ------
        ValueError
            If `base_path` is None.

        Notes
        -----
        Each object is saved in a separate pickle file named '<key>.pkl' inside `base_path`.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        for name, obj in objects.items():
            file_path = os.path.join(self.base_path, f'{name}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        return


    def _create_experiment_dir(self) -> str:
        """
        Create a new experiment directory with progressive ID and timestamp.

        Returns
        -------
        : str
            Path to the created experiment directory.

        Raises
        ------
        ValueError
            If `base_path` is None.
        """
        if self.base_path is None:
            raise ValueError('Cannot save results because `base_path` is None. This saver can only be used for loading data.')

        self.experiment_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(self.base_path, f'experiment_{self.experiment_counter:0{self.padding}}_{timestamp}')

        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
