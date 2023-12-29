
"""
Train and classify from multisource (or project) using featurevector files provided on S3.

"""
import json
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import pickle
import pandas as pd
import io
from operator import itemgetter
from pathlib import Path
from spacer import config
from spacer.data_classes import ImageLabels
from spacer.tasks import (
    train_classifier,
    classify_features,
)
from spacer.messages import (
    DataLocation,
    TrainClassifierMsg,
    ClassifyFeaturesMsg,
)
from spacer.tasks import classify_features, extract_features, train_classifier

## Assign the secrets to boto3

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


## Coralnet

sources = ['s1970', 's2083', 's2170']

## Find the annotations.csv files for the chosen sources

# Create a list for the chosen sources for the annotations f'coralnet_public_features/{source}/annotations.csv'
chosen_sources = []
for source in sources:
    chosen_sources.append(f'coralnet_public_features/{source}/annotations.csv')


#Use S3 to download the annotations.csv files for the chosen sources - check if they exist in the bucket first

# See if chosen_sources are in the s3 bucket using s3_client
bucketname='coralnet-mermaid-share'
for source in chosen_sources:
    try:
        s3_client.head_object(Bucket=bucketname, Key=source)
        print(f"{source} exists in the bucket.")
    except Exception as e:
        print(f"{source} does not exist in the bucket. Error: {e}")


## Append annotations


def read_csv_in_chunks(bucketname, key, chunksize=10000):
    """
    Read a CSV file from S3 in chunks.
    Append the source as a column called 'source_id'
    """

    response = s3_client.get_object(Bucket=bucketname, Key=key)
    lines = []
    header = None
    for line in response['Body'].iter_lines():
        if not header:
            header = line.decode('utf-8')
            continue
        lines.append(line.decode('utf-8'))
        if len(lines) == chunksize:
            chunk = pd.read_csv(io.StringIO(header + '\n' + '\n'.join(lines)))
            chunk['source_id'] = key  # Add the source_id column
            yield chunk
            lines = []
    if lines:
        chunk = pd.read_csv(io.StringIO(header + '\n' + '\n'.join(lines)))
        chunk['source_id'] = key  # Add the source_id column
        yield chunk


appended_df = pd.DataFrame()

for key in chosen_sources:
        first_chunk = True
        for chunk in read_csv_in_chunks(bucketname, key, chunksize=100):
            if first_chunk:
                if appended_df.empty:
                    appended_df = chunk
                else:
                    if not chunk.columns.equals(appended_df.columns):
                        raise ValueError(f"Inconsistent data structure in file: {key}")
                first_chunk = False
            appended_df = pd.concat([appended_df, chunk], ignore_index=True)



# Create a key to download the features from S3 given:
# 'coralnet_public_features/{source_id}/features/i{image_id}'
# E.g. 'coralnet_public_features/s1073/features/i84392.featurevector'
# Where source_id column currently looks like 'coralnet_public_features/s1073/annotations.csv'
# and image_id column currently looks like '84392'
# and the features are stored in the features directory

def format_featurevector_key(image_id, source_id):
    """
    Format the featurevector key to include the directories we need to access in S3.
    E.g. 'coralnet_public_features/{source_id}/features/i{image_id}'
    """
    source_id = source_id.split('/')[1]  # Get the source_id from the source_id column
    image_id = 'i' + str(image_id) + '.featurevector'  # Format the image_id
    return 'coralnet_public_features/' + source_id + '/features/' + image_id


#Apply the format_featurevector_key function to the appended_df dataframe to create a new column called key.


# Format new column in appended_df called key
appended_df['key'] = appended_df.apply(lambda x: format_featurevector_key(x['Image ID'], x['source_id']), axis=1)


## Training Data

# CoralNet uses a 7-to-1 ratio of train_labels to val_labels.
# Calculate the split index
total_labels = len(appended_df)
train_size = int(total_labels * 7 / 8)  # 7 parts for training out of 8 total parts

# Split the data
train_labels_data = appended_df.iloc[:train_size]
val_labels_data = appended_df.iloc[train_size:]


# Convert train_labels_data and val_labels_data to the required format
train_labels_data = {f"{key}": [tuple(x) for x in group[['Row', 'Column', 'Label ID']].values]
                     for key, group in train_labels_data.groupby('key')}


val_labels_data = {f"{key}": [tuple(x) for x in group[['Row', 'Column', 'Label ID']].values]
                   for key, group in val_labels_data.groupby('key')}


## Use spacer to create train classifier msg


train_msg = TrainClassifierMsg(
    job_token='mulitest',
    trainer_name='minibatch',
    nbr_epochs=10,
    clf_type='MLP',
    # A subset
    train_labels=ImageLabels(data = train_labels_data),
    val_labels=ImageLabels(data = val_labels_data),
    #S3 bucketname
    features_loc=DataLocation('s3', bucketname = bucketname, key=''),
    previous_model_locs=[],
    model_loc=DataLocation('filesystem',str(Path.cwd())+'/classifier_test.pkl'),
    valresult_loc=DataLocation('filesystem',str(Path.cwd())+'/valresult_test.json'),

)



return_msg = train_classifier(train_msg)



fileObj = open('return_msg.obj', 'wb')
pickle.dump(return_msg,fileObj)
fileObj.close()


classifier_filepath = Path.cwd() /'classifier_test.pkl'
valresult_filepath = Path.cwd()  / 'valresult_test.json'



ref_accs_str = ", ".join([f"{100*acc:.1f}" for acc in return_msg.ref_accs])

print("------------------------------")
print(f"Classifier stored at: {classifier_filepath}")
print(f"New model's accuracy: {100*return_msg.acc:.1f}%")
print(
    "New model's accuracy progression (calculated on part of train_labels)"
    f" after each epoch of training: {ref_accs_str}")

print(f"Evaluation results:")
with open(valresult_filepath) as f:
    valresult = json.load(f)


## Create matching labels between coralnet and mermaid


label_shortcode = pd.read_csv('coral_net_mermaid_labels.csv')

# Rename ID to class and Default shord code to shortcode
label_shortcode = label_shortcode.rename(columns={'ID': 'classes', 'Default short code': 'shortcode'})

# Get the unique class and shortcode pairs
label_shortcode = label_shortcode[['classes', 'shortcode']].drop_duplicates()




# Create label_list from the label_shortcode
# Account for those classes that don't match in valresult with a default shortcode
label_list = []
for label in valresult['classes']:
    if label in label_shortcode['classes'].values:
        label_list.append(label_shortcode.loc[label_shortcode['classes'] == label, 'shortcode'].iloc[0])
    else:
        label_list.append('unknown')





for ground_truth_i, prediction_i, score in zip(
    valresult['gt'], valresult['est'], valresult['scores']
):
    print(f"Actual = {label_list[ground_truth_i]}, Predicted = {label_list[prediction_i]}, Confidence = {100*score:.1f}%")

print(f"Train time: {return_msg.runtime:.1f} s")




# From chosen_prj, get the features for the source from the features directory from the full Key path
# E.g. coralnet_public_features/s1073/features/i84392.featurevector
# Use Regex to get the features
feature_files = chosen_prj[chosen_prj['Key'].str.contains(r'coralnet_public_features/.*/features/.*\.featurevector$')]




## For each feature file, create a classify features message


# Create a list of classify features messages
# For each feature file, create a classify features message
TOP_SCORES_PER_POINT = 5
messages = []
for key in appended_df['key']:
    message = ClassifyFeaturesMsg(
        job_token=key,
        feature_loc=DataLocation('s3', key=key, bucketname = 'coralnet-mermaid-share'),
        classifier_loc=DataLocation('filesystem', classifier_filepath),
    )
    messages.append(message)
    return_msg = classify_features(message)
    print("------------------------------")
    print(f"Classification result for {key}:")

    label_ids = return_msg.classes
    for i, (row, col, scores) in enumerate(return_msg.scores):
        top_scores = sorted(
            zip(label_ids, scores), key=itemgetter(1), reverse=True)
        top_scores_str = ", ".join([
            f"{[str(label_id)]} = {100*score:.1f}%"
            for label_id, score in top_scores[:TOP_SCORES_PER_POINT]
        ])
        print(f"- Row {row}, column {col}: {top_scores_str}")

    print(f"Classification time: {return_msg.runtime:.1f} s")



