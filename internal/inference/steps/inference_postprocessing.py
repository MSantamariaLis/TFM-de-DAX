import pandas as pd

class InferencePostprocessing:

    """
    Handles postprocessing steps for inference predictions using provided preprocessing objects.

    This class is designed to reverse transformations (such as scaling or differencing)
    that were applied during preprocessing, restoring predictions to their original scale or form.

    Attributes:
        preprocessing_objects (dict): 
            A dictionary mapping preprocessor names (str) to preprocessor objects 
            (e.g., scalers, differencers) that implement an `inverse_transform` method.
    """

    def __init__(self, preprocessing_objects:dict):

        self.preprocessing_objects = preprocessing_objects

    def __apply_step(self, predictions:pd.DataFrame, preprocessor_name: str):

        """ Helper method to apply a specific postprocessing step. """

        preprocessor = self.preprocessing_objects.get(preprocessor_name, None)
        return preprocessor.inverse_transform(predictions) if preprocessor else predictions
    
    def inference_postprocessing(self, predictions:pd.DataFrame):

        """ Applies postprocessing steps to inference predictions to restore them to their original scale or form. """

        predictions = self.__apply_step(predictions, 'difference')
        predictions = self.__apply_step(predictions, 'scaler')

        return predictions


    