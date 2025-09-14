# Identifica las series intermitentes (``sparse``) según el umbral escogido.
class SparsityTest:

    DEFAULTS = {'target_columns': [], 'threshold': 0.25}

    def __init__(self, **kwargs): 
        
        self.__dict__.update(self.DEFAULTS)
        self.__dict__.update(kwargs)
    
    def sparsity_test(self, df):

        sparse = []

        for col in self.target_columns:
            if (df[col] == 0).sum() > df[col].dropna().shape[0] * self.threshold:
                sparse.append(col)
        
        return sparse    