from google.oauth2 import service_account
import pandas as pd
from pandas.io import gbq


# Path to the service account key file
key_path = 'empirical-realm-374307-8bbebe57ceef.json'

# Set up the credentials object
credentials = service_account.Credentials.from_service_account_file(key_path)




# Set the project ID
project_id = 'empirical-realm-374307'
database_name = 'transport_dataset'
table_name = 'product_transport'

# Define SQL query
num_features = ['Net_Weight', 'Value', 'Packaging_Cost', 'Expiry_Period', 'Length', 'Height', 'Width', 'Volume', 'Perishable_Index', 'Flammability_Index', 'F145', 'F7987', 'F992']
labels = ['Air', 'Road', 'Rail', 'Sea']
query = f"""
    SELECT 
        {','.join(num_features)},
        CASE WHEN Size = 'A' THEN 1 ELSE 0 END Size_A,
        CASE WHEN Size = 'B' THEN 1 ELSE 0 END Size_B,
        CASE WHEN Size = 'C' THEN 1 ELSE 0 END Size_C,
        CASE WHEN Size = 'D' THEN 1 ELSE 0 END Size_D,
        {','.join(labels)}
    FROM `{project_id}.{database_name}.{table_name}`"""


processing_queries = []

def read_query(query):
    # Use pandas to execute the query and load the results into a DataFrame
    df = gbq.read_gbq(query=query, project_id=project_id, credentials=credentials)
    return df


