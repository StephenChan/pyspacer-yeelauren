import json
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import os
import pickle
from datetime import datetime, timedelta

# Function to save cache
def save_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Function to load cache
def load_cache(filename, max_age_in_minutes):
    if not os.path.exists(filename):
        return None

    current_time = datetime.now()
    modified_time = datetime.fromtimestamp(os.path.getmtime(filename))
    if current_time - modified_time > timedelta(minutes=max_age_in_minutes):
        return None

    with open(filename, 'rb') as f:
        return pickle.load(f)

# Main script
def main():
    cache_filename = 's3_cache.pkl'
    cache_max_age = 60  # Cache age in minutes

    # Try to load from cache
    cache = load_cache(cache_filename, cache_max_age)
    if cache:
        print("Loading from cache")
        object_count, total_size, folders = cache
    else:
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

            bucketname = 'pyspacer-test'
            paginator = s3_client.get_paginator('list_objects_v2')

            object_count = 0
            total_size = 0
            folders = set()

            for page in paginator.paginate(Bucket=bucketname, Delimiter='/'):
                if 'Contents' in page:
                    object_count += len(page['Contents'])
                    total_size += sum(obj['Size'] for obj in page['Contents'])

                if 'CommonPrefixes' in page:
                    folders.update(prefix['Prefix'] for prefix in page['CommonPrefixes'])

            # Save results to cache
            save_cache((object_count, total_size, folders), cache_filename)

        except (ClientError, BotoCoreError) as e:
            print(f"An AWS error occurred: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"Error reading secrets.json: {e}")
            return
        except IOError as e:
            print(f"File error: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

    print(f"Total objects: {object_count}")
    print(f"Total size (in bytes): {total_size}")

    print("Folders in the bucket:")
    for folder in folders:
        print(folder)

if __name__ == "__main__":
    main()
