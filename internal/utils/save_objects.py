import pickle

def save_pipeline(pipeline, path):

     with open(path + '.pkl', 'wb') as file:
            pickle.dump(pipeline, file)

def load_pipeline(path):
      
    with open(path + '.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    return pipeline
