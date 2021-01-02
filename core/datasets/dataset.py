from pathlib import Path


class Dataset:

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"
    
    def load(self):
        pass
    
    def prepare(self):
        pass