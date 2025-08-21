import os
import pickle

from datetime import datetime
from typing import Any, List, Dict



class ExperimentStorage:
    """
    Class for saving and loading Python objects into experiment-specific folders.

    Each experiment is stored in a directory named:
        experiment_<progressive_id>_<timestamp>

    Objects passed to the saver are serialized individually in this folder.

    Attributes
    ----------
    base_path : str
        Path to the root directory where experiments will be saved.
    experiment_counter : int
        Progressive ID counter for experiments within this storage instance.
    """

    def __init__(self, base_path: str) -> None:
        """
        Initialize the storage with a base directory.

        Parameters
        ----------
        base_path : str
            Path to the root directory where experiments will be saved.
        """
        self.base_path = base_path
        self.experiment_counter = 0

        os.makedirs(self.base_path, exist_ok=True)
        return


    def save_objects(self, objects: List[Any]) -> str:
        """
        Save a list of Python objects into a new experiment folder.

        Parameters
        ----------
        objects : list of Any
            List of objects to save. Each object may have a `name` attribute.
            If not, the Python object ID will be used as filename.

        Returns
        -------
        : str
            Path to the experiment folder where objects were saved.
        """
        exp_dir = self._create_experiment_dir()

        for obj in objects:
            for attr, value in obj.__dict__.items():
                if callable(value) and getattr(value, '__name__', '') == '<lambda>':
                    setattr(obj, attr, None)

            file_name = getattr(obj, 'name', str(id(obj)))
            file_path = os.path.join(exp_dir, f'{file_name}.pkl')

            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        return exp_dir


    def load_objects(self, exp_dir: str) -> Dict[str, Any]:
        """
        Load all saved objects from a given experiment folder.

        Parameters
        ----------
        exp_dir : str
            Path to the experiment directory containing saved objects.

        Returns
        -------
        : dict
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


    def _create_experiment_dir(self) -> str:
        """
        Create a new experiment directory with progressive ID and timestamp.

        Returns
        -------
        : str
            Path to the created experiment directory.
        """
        self.experiment_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(self.base_path, f'experiment_{self.experiment_counter}_{timestamp}')

        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
