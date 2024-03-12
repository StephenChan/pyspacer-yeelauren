"""
Train and classify from multisource (or project) using featurevector files provided on S3.

"""
import argparse
import csv
import io
import json
import logging
import os
import pickle
import shutil
import threading
from datetime import datetime
from typing import Iterable

import boto3
import numpy as np
import psutil
import tqdm
from botocore.exceptions import BotoCoreError, ClientError
from sklearn.model_selection import train_test_split

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


def read_csv_from_s3(input_storage, filepath):
    csv_byte_stream = input_storage.load(filepath)
    # We want text, but load() gives bytes.
    csv_text_stream = io.TextIOWrapper(csv_byte_stream, encoding='utf-8')
    # And for some reason the file pointer's not at the start, so we
    # have to seek there.
    csv_text_stream.seek(0)
    return csv_text_stream


INPUT_BUCKET = 'coralnet-mermaid-share'
INPUT_PATH = 'coralnet_public_features'
OUTPUT_BUCKET = 'coralnet-mermaid-share'
OUTPUT_PATH = 'allsource'


def load_labels_data(included_sources):

    input_storage = storage_factory('s3', INPUT_BUCKET)

    if included_sources:
        source_ids = included_sources
    else:
        # All sources from sources.csv
        source_ids = []
        with read_csv_from_s3(
                input_storage, f'{INPUT_PATH}/sources.csv') as sources_csv:
            sources_reader: Iterable[dict] = csv.DictReader(sources_csv)
            for row in sources_reader:
                source_ids.append(int(row["Source ID"]))

    labels_data = dict()

    for source_id in tqdm.tqdm(source_ids):

        annotations_filepath = f'{INPUT_PATH}/s{source_id}/annotations.csv'

        if not input_storage.exists(annotations_filepath):
            continue

        with read_csv_from_s3(
                input_storage, annotations_filepath) as annotations_csv:

            image_id = None
            annotations_reader: Iterable[dict] = csv.DictReader(
                annotations_csv)

            for row in annotations_reader:
                if image_id != row["Image ID"]:
                    # End of previous image's annotations, and start of
                    # another image's annotations.
                    image_id = row["Image ID"]
                    feature_filepath = (
                        f'{INPUT_PATH}/s{source_id}/features/'
                        f'i{image_id}.featurevector'
                    )

                    labels_data[feature_filepath] = []

                annotation = (
                    int(row["Row"]),
                    int(row["Column"]),
                    int(row["Label ID"]),
                )
                labels_data[feature_filepath].append(annotation)

    return labels_data


def write_bytestream_from_s3(output_storage, s3_filepath, out_stream):
    in_stream = output_storage.load(s3_filepath)
    in_stream.seek(0)
    shutil.copyfileobj(in_stream, out_stream)


parser = argparse.ArgumentParser(
    description="Runs on all sources, or only the ones"
                " specified by --included_sources.")
parser.add_argument(
    '--included_sources', type=int, nargs='+',
    help="List of source IDs to include in the classifier training.")
args = parser.parse_args()

logger.info('Setting up connections')
log_memory_usage('Initial memory usage')
# Set up connections
# Set up Sources
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
bucketname = "coralnet-mermaid-share"
prefix = "coralnet_public_features/"

labels_data = load_labels_data(args.included_sources)

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
