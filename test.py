import csv
import json
from operator import itemgetter
import os
from datetime import datetime
from botocore.exceptions import NoCredentialsError, ClientError
from pathlib import Path
from spacer import config
from scripts.docker import runtimes
from spacer.tasks import process_job
from spacer.storage import load_image, store_image
from spacer.messages import JobMsg, DataLocation, ExtractFeaturesMsg
from spacer.extract_features import EfficientNetExtractor
from spacer.messages import (
    ClassifyFeaturesMsg,
    DataLocation,
    ExtractFeaturesMsg,
    TrainClassifierMsg,
)
from spacer.tasks import classify_features, extract_features, train_classifier

# Load the secret.json file
with open('secrets.json', 'r') as f:
    secrets = json.load(f)
# Use docker/runtimes.py to cr
nbr_rowcols = 10
image_size = 1000
image_key='images-annotated/23_7168.JPG'
extractor = 'efficientnet_b0_ver1'

org_img_loc = DataLocation(storage_type='s3',
                               key=image_key,
                               bucketname=config.TEST_BUCKET)

org_img = load_image(org_img_loc)