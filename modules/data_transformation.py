import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def scale_data(data_refined):

    column_names = list(data_refined.columns.tolist())
    binary_indices = column_names[0:3] 
    numeric_indices = column_names[3:14] 

    # ColumnTransformer using indices
    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", "passthrough", binary_indices),    
            ("num", StandardScaler(), numeric_indices)
            ]
        )

    # Fit & transform
    transformed_df = pd.DataFrame(
        preprocessor.fit_transform(data_refined),
        columns=data_refined.columns,
        index=data_refined.index
    )
    
    # Preserve column names
    scaled_df = pd.DataFrame(
        transformed_df , columns= binary_indices + numeric_indices,
        index=data_refined.index)
    
    return scaled_df, preprocessor

