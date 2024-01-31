"""
Train and classify from multisource (or project) using featurevector files provided on S3 from a subset of sources

"""
import json
import os
import pickle
import s3fs
import boto3
import duckdb
import logging 
import psutil
import threading
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from pyarrow import fs

from spacer.data_classes import ImageLabels
from spacer.messages import (
    DataLocation,
    TrainClassifierMsg,
)
from spacer.tasks import (
    train_classifier,
)
from spacer.task_utils import preprocess_labels

load_dotenv()
# Set up logging 
logging.basicConfig(
    filename='app.log',
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
# Set up Sources
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
bucketname = "coralnet-mermaid-share"
prefix = "coralnet_public_features/"

# Use Pandas and s3fs to read the csv from s3
s3 = s3fs.S3FileSystem(
    anon=False,
    use_ssl=True,
    client_kwargs={
        "region_name": os.environ['S3_REGION'],
        "endpoint_url": os.environ['S3_ENDPOINT'],
        "aws_access_key_id": os.environ['S3_ACCESS_KEY'],
        "aws_secret_access_key": os.environ['S3_SECRET_KEY'],
        "verify": True,
    }
)
# Use pyarrow to read the parquet file from S3
fs = fs.S3FileSystem(
    region=os.environ["S3_REGION"],
    access_key=os.environ["S3_ACCESS_KEY"],
    secret_key=os.environ["S3_SECRET_KEY"],
)

#sources_to_keep = pd.read_csv(s3.open('allsource/list_of_sources.csv', mode='rb'))
# Construct the sources into a df to feed into existing code 
sources_to_keep = pd.DataFrame(
    {
        "source_id": ['s1970', 's2083', 's2170']
    }
)
# TODO check this is the most efficient way to load in parquet
# Load Parquet file from S3 bucket using s3_client
parquet_file = "pyspacer-test/allsource/CoralNet_Annotations_SourceID.parquet"
logger.info('Reading parquet file from S3 bucket')
log_memory_usage('Memory usage after reading parquet file')

df = pq.read_table(parquet_file, filesystem=fs)
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


# Rewrite
labels_data = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in selected_sources.groupby("key")
}


log_memory_usage('Memory usage after creating train_labels_data and val_labels_data')
logger.info('Create TrainClassifierMsg')

train_msg = TrainClassifierMsg(
    job_token="mulitest",
    trainer_name="minibatch",
    nbr_epochs=1,
    clf_type="MLP",
    # A subset
    labels=ImageLabels(labels_data),
    features_loc=DataLocation("s3", bucketname=bucketname, key=""),
    previous_model_locs=[],
    model_loc=DataLocation(
        "s3", bucketname="pyspacer-test", key="allsource" + f"/classifier_all_source_{current_time}.pkl"
    ),
    valresult_loc=DataLocation(
        "s3", bucketname="pyspacer-test", key="allsource" + f"/valresult_all_source_{current_time}.json"
    ),
)

logger.info('Train Classifier')
log_memory_usage('Memory usage before training')
return_msg = train_classifier(train_msg)
logger.info(f'Train time: {return_msg.runtime:.1f} s')
log_memory_usage('Memory usage after training')
logger.info('Write return_msg to S3')

path = f's3://pyspacer-test/allsource/return_msg_all_source_{current_time}.pkl'

# Use pickle to serialize the object
return_msg_bytes = pickle.dumps(return_msg)

# Use s3fs to write the bytes to the file
with s3.open(path, 'wb') as f:
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