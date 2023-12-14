import pandas as pd
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import numpy as np
import logging 
import psutil
import boto3
import json
from botocore.exceptions import ClientError, BotoCoreError
import duckdb
import os
from datetime import datetime
from pyarrow import fs
import pyarrow.parquet as pq
from spacer.data_classes import ImageLabels
from spacer.messages import (
    DataLocation,
    TrainClassifierMsg,
)
from spacer.tasks import (
    train_classifier,
)

cwd = os.getcwd()
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
bucketname = 'coralnet-mermaid-share'
prefix = 'coralnet_public_features/'

# Construct the sources into a df to feed into existing code
sources_to_keep = pd.DataFrame(
    {
        "source_id": ['s1970', 's2083', 's2170']
    }
)
# Set up logging
logging.basicConfig(
    filename=f'log_{current_time}.log',
     filemode='w',
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
def log_memory_usage(message):
    memory_usage = psutil.virtual_memory()
    logger.info(f"{message} - Memory usage: {memory_usage.percent}%")

logger = logging.getLogger(__name__)  # Create a logger for your script

logger.info('Setting up connections')
log_memory_usage('Initial memory usage')
# Set up connections
try:
    # Load the secret.json file
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)

    # Create a session using the credentials from secrets.json
    s3_client = boto3.client(
        's3',
        region_name=secrets['AWS_REGION'],
        aws_access_key_id=secrets['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=secrets['AWS_SECRET_ACCESS_KEY']
    )
except (ClientError, BotoCoreError) as e:
    print(f"An AWS error occurred: {e}")
except json.JSONDecodeError as e:
    print(f"Error reading secrets.json: {e}")
except IOError as e:
    print(f"File error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Read Parquet File

parquet_file = os.path.join(cwd, 'CoralNet_Annotations_SourceID.parquet')

# Read the Parquet file into a Pandas DataFrame
df = pd.read_parquet(parquet_file)

# Connect to DuckDB
conn = duckdb.connect()

# Create a DuckDB table from the DataFrame
duckdb_df = conn.from_df(df)

logger.info('Reading parquet file from local dir')
log_memory_usage('Memory usage after reading parquet file')

df = pq.read_table(parquet_file)
# Connect to DuckDB
logger.info('Connecting to DuckDB')
conn = duckdb.connect()
# Create a DuckDB table
logger.info('Creating DuckDB table')
conn.execute("CREATE TABLE duckdb_df AS SELECT * FROM df")
logger.info('Wranging in DuckDB based on sources_to_keep')
# Create the feature vector parsing in duckdb
# Add a new column 'key' to the table
# Update the 'key' column with the desired values
conn.execute(
    f"""
    CREATE TABLE selected_sources AS 
    SELECT *, 'coralnet_public_features/' || CAST(source_id AS VARCHAR) || '/features/i' || CAST("Image ID" AS VARCHAR) || '.featurevector' AS key
    FROM duckdb_df 
    WHERE "source_id" IN {tuple(sources_to_keep["source_id"].tolist())}
    """
)

selected_sources = conn.execute("SELECT * FROM selected_sources").fetchdf()
log_memory_usage('Memory usage after wrangling in DuckDB')
# Training based on CoralNet's ratio
# TODO: This is a good place to start, but we should also consider the ratio of images per label.

logger.info('Split Training/ Val based on CoralNet\'s ratio')
total_labels = len(selected_sources)
train_size = int(total_labels * 7 / 8)

logger.info('Create train_labels_data and val_labels_data')
# Use duckdb to create train_labels_data and val_labels_data
train_labels_data = conn.execute(
    f"SELECT * FROM selected_sources LIMIT {train_size}"
).fetchdf()

val_labels_data = conn.execute(
    f"SELECT * FROM selected_sources OFFSET {train_size} ROWS"
).fetchdf()

logger.info('Restructure as Tuples for ImageLabels')

# Rewrite
train_labels_data = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in train_labels_data.groupby("key")
}


val_labels_data = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in val_labels_data.groupby("key")
}
log_memory_usage('Memory usage after creating train_labels_data and val_labels_data')
logger.info('Create TrainClassifierMsg')

train_msg = TrainClassifierMsg(
    job_token="mulitest",
    trainer_name="minibatch",
    nbr_epochs=1,
    clf_type="MLP",
    # A subset
    train_labels=ImageLabels(data=train_labels_data),
    val_labels=ImageLabels(data=val_labels_data),
    # S3 bucketname
    features_loc=DataLocation("s3", bucketname=bucketname, key=""),
    previous_model_locs=[],
    model_loc=DataLocation(
        "filesystem",
        f"/classifier_subset_source_{current_time}.pkl"
    ),
    valresult_loc=DataLocation(
        "filesystem",
      f"/valresult_subset_source_{current_time}.json"
    ),
)

logger.info('Train Classifier')
log_memory_usage('Memory usage before training')
return_msg = train_classifier(train_msg)
logger.info(f'Train time: {return_msg.runtime:.1f} s')
log_memory_usage('Memory usage after training')
logger.info('Write return_msg to local')

path = f'./return_msg_subset_source_{current_time}.pkl'

# Use pickle to serialize the object
return_msg_bytes = pickle.dumps(return_msg)

with open(path, 'wb') as f:
    f.write(return_msg_bytes)


ref_accs_str = ", ".join([f"{100*acc:.1f}" for acc in return_msg.ref_accs])

print("------------------------------")

print(f"New model's accuracy: {100*return_msg.acc:.1f}%")
print(
    "New model's accuracy progression (calculated on part of train_labels)"
    f" after each epoch of training: {ref_accs_str}"
)
logger.info('Done --------- Closing DuckDB connection')
duckdb.close(conn)
