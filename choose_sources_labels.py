import json
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta

bucketname = 'coralnet-mermaid-share'
prefix = 'coralnet_public_features/'
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

def list_folders(s3_client, bucketname, prefix):
    """
    List all folders in an S3 bucket under a specific prefix.

    Args:
        s3_client (boto3.client): The S3 client.
        bucketname (str): The name of the S3 bucket.
        prefix (str): The prefix (folder path).

    Returns:
        list: A list of folder names.

    Example:
        s3_client = boto3.client('s3', region_name='us-west-2')
        bucketname = 'my-bucket'
        prefix = 'my-folder/'
        folders = list_folders(s3_client, bucketname, prefix)
        print(folders)
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    folders = []
    for page in paginator.paginate(Bucket=bucketname, Delimiter='/', Prefix=prefix):
        for prefix in page.get('CommonPrefixes', []):
            folders.append(prefix['Prefix'])
    return folders


folders = list_folders(s3_client, bucketname, prefix)

# For a given folder, filter out specific keys that are not needed
def filter_keys(s3_client, bucketname, prefix, keys):
    """
    Filter out specific keys that are not needed.

    Args:
        s3_client (boto3.client): The S3 client.
        bucketname (str): The name of the S3 bucket.
        prefix (str): The prefix (folder path).
        keys (list): A list of keys to filter out.

    Returns:
        list: A list of keys.

    Example:
        s3_client = boto3.client('s3', region_name='us-west-2')
        bucketname = 'my-bucket'
        prefix = 'my-folder/'
        keys = ['my-folder/file1.txt', 'my-folder/file2.txt']
        keys = filter_keys(s3_client, bucketname, prefix, keys)
        print(keys)
    """
    filtered_keys = []
    for key in keys:
        if not key.endswith('/'):
            filtered_keys.append(key)
    return filtered_keys

# Group appemded dataframe by class and total the amount

# Group by source and class

# Group by number of points per source


