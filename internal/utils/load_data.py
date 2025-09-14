import pandas as pd
import os
import pickle
import json

class LocalDataManager:
     
    def __init__(self,
                 client_name: str,
                 file_format: str = 'csv',
                 sep: str = ',',
                 ):
        
        self.client_name = client_name   
        self.sep = sep

        self.file_format = file_format
        self.folder_path = os.path.join("..", "predictorlis", "examples", self.client_name)

    def __data_reader(self, data_path):
        """Helper method to read df from a given path. """

        reader_dict = {
            'csv' : pd.read_csv,
            'parquet': pd.read_parquet,
            'xlsx': pd.read_parquet,
        }

        if self.file_format in reader_dict: return reader_dict[self.file_format](data_path, sep = self.sep)
        else: raise NameError(f'The format: {self.file_format} is not supported. Available options are: {list(reader_dict.keys())}')

    def read_data(self, phase: str = 'train', covariates: bool = False ):

        df_to_read = f'df_{phase}_covariates' if covariates else f'df_{phase}'
        data_path = os.path.join(self.folder_path, "datasets", phase, f"{df_to_read}.{self.file_format}")
        
        df = self.__data_reader(data_path) if os.path.exists(data_path) else FileNotFoundError('The path does not exit. ')
        return df
    
    def read_json(self):

        config_path = os.path.join(self.folder_path, "config", "config.json")

        with open(config_path) as json_file:
            json_data = json.load(json_file)
         
        return json_data
    
    def save_pipeline(self, pipeline:object, name:str):

        """ Saves a pipeline as a pickle object """
        
        pipeline_path = os.path.join(self.folder_path, "objects", f"{name}.pkl")

        with open(pipeline_path, 'wb') as file:
                pickle.dump(pipeline, file)
          
    def load_pipeline(self, name):

        """ Loads pipeline"""

        pipeline_path = os.path.join(self.folder_path, "objects", f"{name}.pkl")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

        with open(pipeline_path + '.pkl', 'rb') as file:
            pipeline = pickle.load(file)

        return pipeline