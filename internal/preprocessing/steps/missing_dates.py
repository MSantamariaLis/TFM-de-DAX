import pandas as pd

#TODO: change this so the dates of the covariate(s) ts match the dates of the target ts

"""
calculate ->

ids_date_range : {'ID-1.1': {'1-1-2020': '1-12-2024'}}
"""

class MissingDates:

    def __init__(self, ids_col, date_col, label_col,  **kwargs):
        
        self.ids_col = ids_col
        self.date_col = date_col
        
        self.label_col = label_col
        self.frequency = kwargs.get('frequency')
        
        self.ids_date_range = {}

    def group_fit_transform(self, df_grouped:pd.DataFrame, end_date:str):

        unique_id = df_grouped[self.ids_col].iloc[0]
        start_date = df_grouped[self.date_col].min()

        self.ids_date_range[unique_id] = {'start_date': start_date, 'end_date': end_date}
        df_grouped = df_grouped.drop_duplicates(subset=[self.date_col])

        df_grouped = df_grouped.sort_values(self.date_col).drop_duplicates(subset=[self.date_col])       
        new_range = pd.date_range(start=start_date, end=end_date, freq=self.frequency)
        
        df_grouped = df_grouped.set_index(self.date_col).reindex(new_range).reset_index().rename(columns={'index': self.date_col})
        df_grouped[self.ids_col] = df_grouped[self.ids_col].ffill().bfill()

        return df_grouped.infer_objects()
    
    def group_fit_transform_covariates(self, df_grouped_covariates:pd.DataFrame):

        unique_id = df_grouped_covariates[self.ids_col].iloc[0]
        reduced_date_range = self.ids_date_range.get(unique_id)

        start_date = reduced_date_range['start_date']
        end_date = reduced_date_range['end_date']

        new_range = pd.date_range(start=start_date, end=end_date, freq=self.frequency)
        df_grouped_covariates = df_grouped_covariates.set_index(self.date_col).reindex(new_range).reset_index().rename(columns={'index': self.date_col})

        df_grouped_covariates[self.ids_col] = df_grouped_covariates[self.ids_col].ffill().bfill()
        return df_grouped_covariates.infer_objects()
    
    def group_transform(self, df_predict_grouped:pd.DataFrame,start_date:str, end_date: str, freq:str):

        """
        create a date range for the dataset to predict. 
        from the last day to predict + n_steps in the future. 

        """
      
        new_range = pd.date_range(start=start_date, end = end_date, freq=freq)
        df_predict_grouped = df_predict_grouped.set_index(self.date_col).reindex(new_range).reset_index().rename(columns={'index': self.date_col})

        df_predict_grouped[self.ids_col] = df_predict_grouped[self.ids_col].ffill().bfill()
        return df_predict_grouped

    def group_transform_covariates(self, df_grouped_covariates:pd.DataFrame,start_date:str, end_date: str, freq:str): 

        new_range = pd.date_range(start=start_date, end = end_date, freq=freq)
        df_grouped_covariates = df_grouped_covariates.set_index(self.date_col).reindex(new_range).reset_index().rename(columns={'index': self.date_col})

        df_grouped_covariates[self.ids_col] = df_grouped_covariates[self.ids_col].ffill().bfill()
        return df_grouped_covariates


    def fit_transform(self, df:pd.DataFrame):

        end_date = df[self.date_col].max()
        df_clean = (
                    df.groupby(self.ids_col, group_keys=False)
                    .apply(self.group_fit_transform, 
                           end_date = end_date))

        return df_clean
    
    def fit_transform_covariates(self, df_covariates:pd.DataFrame):

        df_covariates_clean = (
            df_covariates.groupby(self.ids_col, group_keys=False)
            .apply(self.group_fit_transform_covariates))
        
        return df_covariates_clean
    
    def transform(self, df_predict: pd.DataFrame):

        start_date = df_predict[self.date_col].min()
        end_date = df_predict[self.date_col].max()

        df_predict_clean = (
                    df_predict.groupby(self.ids_col, group_keys=False)
                    .apply(self.group_transform, 
                           start_date = start_date,
                           end_date = end_date,
                           freq = self.frequency))

        return df_predict_clean        


    def transform_covariates(self, df_covariates_future):
        
        start_date = df_covariates_future[self.date_col].min()
        end_date = df_covariates_future[self.date_col].max()

        df_covariates_clean = (
                    df_covariates_future.groupby(self.ids_col, group_keys=False)
                    .apply(self.group_transform_covariates, 
                           start_date = start_date,
                           end_date = end_date,
                           freq = self.frequency))

        return df_covariates_clean



if __name__  == '__main__':

    import pandas as pd

    ids_col, date_col, label_col =  'KeyCIF', 'KeyFecha', 'DiasPerdidos'

    df = pd.read_csv('examples/Mutua/df_train.csv').rename(columns = {date_col: 'ds', ids_col:'unique_id', label_col: 'y'}).sort_values(by= ['unique_id', 'ds'])
    df_covariates = pd.read_csv('examples/Mutua/df_cov_train.csv').rename(columns = {date_col: 'ds', ids_col:'unique_id'}).sort_values(by= ['unique_id', 'ds'])

    df_past_covariates = pd.read_csv('examples/Mutua/df_cov_test.csv').rename(columns = {date_col: 'ds', ids_col:'unique_id'}).sort_values(by= ['unique_id', 'ds'])
    df_future_covariates = pd.read_csv('examples/Mutua/df_cov_forecast.csv').rename(columns = {date_col: 'ds', ids_col:'unique_id'}).sort_values(by= ['unique_id', 'ds'])


    # # # # # # # # #
    #
    # # # # # # # # # 

    fraction_to_drop = 0.01
    rows_to_drop = df.sample(frac=fraction_to_drop, random_state=42).index
    df = df.drop(index=rows_to_drop)

    rows_to_drop = df_covariates.sample(frac=fraction_to_drop, random_state=42).index
    df_covariates = df_covariates.drop(index=rows_to_drop)



    # df = df[df['ds'] != '2019-02-01']
    # print(df)

    df['ds'] = pd.to_datetime(df['ds'])
    df_covariates['ds'] = pd.to_datetime(df_covariates['ds'])

    print(df_covariates)
 
    transformer = MissingDates(ids_col='unique_id', date_col='ds', label_col='y', frequency = 'MS')

    df_cleaned = transformer.fit_transform(df)
    df_covarites_cleaned = transformer.fit_transform_covariates(df_covariates)


    print(df_covarites_cleaned)





         



