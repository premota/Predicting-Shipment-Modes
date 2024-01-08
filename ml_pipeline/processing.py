from .utils import *
from sklearn import model_selection






# Define function to process raw data into a processed dataset
def read_and_process_data():
    
    # Execute the query and load the results into a Pandas dataframe
    processed_data = read_query(query)

    x = processed_data.drop(columns=labels).astype("float32")
    y = processed_data[labels]

    cv_folds = list(model_selection.KFold(n_splits=5, shuffle=True, random_state=42).split(x))
    
    return processed_data, x, y, cv_folds






