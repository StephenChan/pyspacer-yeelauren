"""
Train and classify from multisource (or project) using featurevector files provided on S3.

"""
import json
import os
import pickle

import boto3
import duckdb
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

load_dotenv()
# Set up connections
try:
    # Create a session using the credentials from dotenv
    s3_client = boto3.client(
        "s3",
        region_name=os.environ["S3_REGION"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )
except (ClientError, BotoCoreError) as e:
    print(f"An AWS error occurred: {e}")
except json.JSONDecodeError as e:
    print(f"Error reading secrets.json: {e}")
except IOError as e:
    print(f"File error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Set up Sources
bucketname = "coralnet-mermaid-share"
prefix = "coralnet_public_features/"
sources_to_keep = pd.read_csv("list_of_sources.csv")

# Sources using s3fs

# Use pyarrow to read the parquet file from S3
fs = fs.S3FileSystem(
    region=os.environ["S3_REGION"],
    access_key=os.environ["S3_ACCESS_KEY"],
    secret_key=os.environ["S3_SECRET_KEY"],
)
# TODO check this is the most efficient way to load in parquet
# Load Parquet file from S3 bucket using s3_client
parquet_file = "pyspacer-test/allsource/CoralNet_Annotations_SourceID.parquet"

df = pq.read_table(parquet_file, filesystem=fs)
# Connect to DuckDB
conn = duckdb.connect()
# Create a DuckDB table

conn.execute("CREATE TABLE duckdb_df AS SELECT * FROM df")
# Since we have this in duckdb let's restructure some of the training code in hopes that it also benefits from significant speedups in pre-processing.

# Create the feature vector parsing in duckdb
# Add a new column 'key' to the table
conn.execute("ALTER TABLE duckdb_df ADD COLUMN key VARCHAR")

# Update the 'key' column with the desired values
conn.execute(
    """
    UPDATE duckdb_df 
    SET key = 'coralnet_public_features/' || CAST(source_id AS VARCHAR) || '/features/i' || CAST("Image ID" AS VARCHAR) || '.featurevector'
    """
)

# Now you can use the updated duckdb_df in your queries
conn.execute(
    f"""
    CREATE TABLE selected_sources AS 
    SELECT * FROM duckdb_df WHERE "source_id" IN {tuple(sources_to_keep["source_id"].tolist())}
    """
)

selected_sources = conn.execute("SELECT * FROM selected_sources").fetchdf()
# Training based on CoralNet's ratio
# TODO: This is a good place to start, but we should also consider the ratio of images per label.
total_labels = len(selected_sources)
train_size = int(total_labels * 7 / 8)

# Use duckdb to create train_labels_data and val_labels_data
train_labels_data = conn.execute(
    f"SELECT * FROM selected_sources LIMIT {train_size}"
).fetchdf()

val_labels_data = conn.execute(
    f"SELECT * FROM selected_sources OFFSET {train_size} ROWS"
).fetchdf()

# Rewrite
train_labels_data = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in train_labels_data.groupby("key")
}


val_labels_data = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in val_labels_data.groupby("key")
}


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
        "s3", bucketname="pyspacer-test", key="allsource" + "/classifier_all_source.pkl"
    ),
    valresult_loc=DataLocation(
        "s3", bucketname="pyspacer-test", key="allsource" + "/valresult_all_source.json"
    ),
)

return_msg = train_classifier(train_msg)


print(f"Train time: {return_msg.runtime:.1f} s")

return_msg_bytes = pickle.dumps(return_msg)
s3_client.put_object(
    Bucket="pyspacer-test",
    Key="allsource" + "/return_msg_all_source.pkl",
    Body=return_msg_bytes,
)

ref_accs_str = ", ".join([f"{100*acc:.1f}" for acc in return_msg.ref_accs])

print("------------------------------")

print(f"New model's accuracy: {100*return_msg.acc:.1f}%")
print(
    "New model's accuracy progression (calculated on part of train_labels)"
    f" after each epoch of training: {ref_accs_str}"
)

duckdb.close(conn)
