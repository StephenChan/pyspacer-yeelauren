"""
Train and classify from multisource (or project) using featurevector files provided on S3.

"""
import json
import logging
import os
import pickle
import shutil
import threading
from datetime import datetime

import boto3
import duckdb
import numpy as np
import psutil
import pyarrow.parquet as pq
from botocore.exceptions import BotoCoreError, ClientError
from pyarrow import fs
from sklearn.model_selection import train_test_split

from spacer import config
from spacer.data_classes import ImageLabels
from spacer.messages import DataLocation, TrainClassifierMsg, TrainingTaskLabels
from spacer.storage import storage_factory
from spacer.task_utils import preprocess_labels
from spacer.tasks import (
    train_classifier,
)


# Set up logging 
logging.basicConfig(
    filename='app_all_source_la.log',
     filemode='w',
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
def log_memory_usage(message):
    memory_usage = psutil.virtual_memory()
    logger.info(f"{message} - Memory usage: {memory_usage.percent}%")

logger = logging.getLogger(__name__)  # Create a logger for your script


OUTPUT_BUCKET = 'coralnet-mermaid-share'
OUTPUT_PATH = 'allsource'


def write_bytestream_from_s3(output_storage, s3_filepath, out_stream):
    in_stream = output_storage.load(s3_filepath)
    in_stream.seek(0)
    shutil.copyfileobj(in_stream, out_stream)


logger.info('Setting up connections')
log_memory_usage('Initial memory usage')
# Set up connections
# Set up Sources
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
bucketname = "coralnet-mermaid-share"
prefix = "coralnet_public_features/"

# Use pyarrow to read the parquet file from S3
fs = fs.S3FileSystem(
    region=config.AWS_REGION,
    access_key=config.AWS_ACCESS_KEY_ID,
    secret_key=config.AWS_SECRET_ACCESS_KEY,
)

# TODO check this is the most efficient way to load in parquet
# Load Parquet file from S3 bucket using s3_client
parquet_file = (
    "coralnet-mermaid-share"
    "/multi_source_classifier/selected_sources_labels.parquet")
logger.info('Reading parquet file from S3 bucket')
log_memory_usage('Memory usage after reading parquet file')

selected_sources = pq.read_table(parquet_file, filesystem=fs)
# Connect to DuckDB
logger.info('Connecting to DuckDB')
conn = duckdb.connect()

selected_sources = conn.execute("SELECT * FROM selected_sources").fetchdf()
log_memory_usage('Memory usage after wrangling in DuckDB')
#
# Remove label counts lower than 1
count_labels = selected_sources["Label ID"].value_counts()
#selected_sources = selected_sources[selected_sources["Label ID"].isin(count_labels[count_labels > 1].index)]

# Create a dictionary of the labels to go into existing pyspacer code
logger.info('Restructure as Tuples for ImageLabels')

labels_data = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in selected_sources.groupby("key")
}

log_memory_usage('Memory usage after creating train_labels_data and val_labels_data')
logger.info('Create TrainClassifierMsg')

output_storage = storage_factory('s3', OUTPUT_BUCKET)
classifier_filename = f'classifier_all_source_{current_time}.pkl'
classifier_filepath = f'{OUTPUT_PATH}/{classifier_filename}'
valresult_filename = f'valresult_all_source_{current_time}.json'
valresult_filepath = f'{OUTPUT_PATH}/{valresult_filename}'

train_msg = TrainClassifierMsg(
    job_token="mulitest",
    trainer_name="minibatch",
    nbr_epochs=10,
    clf_type="MLP",
    labels=ImageLabels(labels_data),
    features_loc=DataLocation("s3", bucket_name=bucketname, key=""),
    previous_model_locs=[],
    model_loc=DataLocation(
        "s3", bucket_name=OUTPUT_BUCKET, key=classifier_filepath
    ),
    valresult_loc=DataLocation(
        "s3", bucket_name=OUTPUT_BUCKET, key=valresult_filepath
    ),
)
logger.info('Train Classifier')
log_memory_usage('Memory usage before training')
print('Training Classifier')
return_msg = train_classifier(train_msg)

logger.info(f'Train time: {return_msg.runtime:.1f} s')
log_memory_usage('Memory usage after training')
logger.info('Write return_msg to S3')

# Download results locally and delete from S3
write_bytestream_from_s3(
    output_storage, classifier_filepath, open(classifier_filename, 'wb'))
write_bytestream_from_s3(
    output_storage, valresult_filepath, open(valresult_filename, 'wb'))
output_storage.delete(classifier_filepath)
output_storage.delete(valresult_filepath)

print(f"Train time: {return_msg.runtime:.1f} s")

ref_accs_str = ", ".join([f"{100*acc:.1f}" for acc in return_msg.ref_accs])

print("------------------------------")

print(f"New model's accuracy: {100*return_msg.acc:.1f}%")
print(
    "New model's accuracy progression (calculated on part of train_labels)"
    f" after each epoch of training: {ref_accs_str}"
)
