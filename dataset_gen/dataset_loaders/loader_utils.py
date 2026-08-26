from abc import ABC, abstractmethod
from typing import Generator
import sys
import os
import importlib
import inspect
from typing import Type
from dotenv import load_dotenv

#holds the dataset root path
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config.env'))

class Loader(ABC):
    @abstractmethod
    def sample_loader(self, split:str, target_size: int, index: int=0) -> Generator[str, None, None]:
        """
        Abstract generator method that should yield strings.
        :param index: Optional starting index
        """
        pass
    def get_class_name(self) -> str:
        """Return the class name of the actual subclass."""
        return type(self).__name__
    
    

    
def find_loader_classes() ->list[Type[Loader]]:
    # Determine the directory of this script (loader_base.py)
    directory = os.path.dirname(os.path.abspath(__file__))
    loaders = []

    # Add directory to sys.path to allow sibling imports
    if directory not in sys.path:
        sys.path.insert(0, directory)

    # Scan only the files in this directory
    for filename in os.listdir(directory):
        if filename.endswith("loader.py") and filename != os.path.basename(__file__):
            module_path = os.path.join(directory, filename)
            module_name = os.path.splitext(filename)[0]

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            # try:
            spec.loader.exec_module(module)
            # except Exception as e:
            #     print(f"Error importing {module_name}: {e}")
            #     continue

            # Inspect the module for class definitions
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure class is defined in this module
                if obj.__module__ == module_name:
                    loaders.append(obj)

    return loaders