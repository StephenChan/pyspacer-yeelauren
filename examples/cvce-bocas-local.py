import csv
import json
from operator import itemgetter
import os
from pathlib import Path
from spacer import config
from spacer.data_classes import ImageLabels
from spacer.extract_features import EfficientNetExtractor
from spacer.messages import (
    ClassifyFeaturesMsg,
    DataLocation,
    ExtractFeaturesMsg,
    TrainClassifierMsg,
)
from spacer.tasks import classify_features, extract_features, train_classifier


TOP_SCORES_PER_POINT = 5

# Use S3 as the DataLocation and configure the AWS credentials
# Use secrets.json to retrieve the AWS credentials
# Use the AWS credentials to configure the DataLocation
data_loc = DataLocation(
    storage_type='s3',
    key=config.AWS_SECRET_ACCESS_KEY,
    bucketname=config.TEST_BUCKET
)

# Add tests to check connection
# Add tests to check if the bucket exists
# Add tests to check if the bucket is empty


module_dir = Path(__file__).resolve().parentDataLocation
output_dir = module_dir / 'output'


def image_to_feature_filepath(image_filepath: Path) -> Path:
    return output_dir / 'features' / (image_filepath.stem + '.featurevector')


def image_filepath_to_id(image_filepath: Path) -> str:
    # Image IDs aren't a pyspacer construct; this is just how this
    # example organizes the images and annotations.
    return image_filepath.stem.split('_')[-1]


if __name__ == '__main__':

    annotated_image_dir = module_dir / 'images-annotated'
    unannotated_image_dir = module_dir / 'images-unannotated'
    extractor_weights_filepath = module_dir / 'efficientnet_b0_ver1.pt'

    with open(module_dir / 'annotations.json') as f:
        all_annotations = json.load(f)

    label_ids_to_codes = dict()
    with open(module_dir / 'labelset.csv') as f:
        reader = csv.reader(f)
        for label_id, label_code in reader:
            label_ids_to_codes[label_id] = label_code

    annotated_image_filepaths = []
    # Sort for predictable ordering
    for filename in sorted(os.listdir(annotated_image_dir)):
        annotated_image_filepaths.append(annotated_image_dir / filename)
    unannotated_image_filepaths = []
    for filename in sorted(os.listdir(unannotated_image_dir)):
        unannotated_image_filepaths.append(unannotated_image_dir / filename)
    all_image_filepaths = (
        annotated_image_filepaths + unannotated_image_filepaths)

    # Extract features

    print("------------------------------")

    for image_filepath in all_image_filepaths:
        feature_filepath = image_to_feature_filepath(image_filepath)
        image_id = image_filepath_to_id(image_filepath)
        annotations = all_annotations[image_id]

        message = ExtractFeaturesMsg(
            job_token=image_filepath.name,
            extractor=EfficientNetExtractor(
                data_locations=dict(
                    weights=DataLocation(
                        'filesystem', str(extractor_weights_filepath)),
                ),
            ),
            rowcols=[(row, col) for row, col, _ in annotations],
            image_loc=DataLocation('filesystem', str(image_filepath)),
            feature_loc=DataLocation('filesystem', str(feature_filepath)),
        )
        return_message = extract_features(message)
        print(
            f"Feature vector stored at: {feature_filepath}"
            f" (extract time: {return_message.runtime:.1f} s)")

    # Train classifier

    train_labels_data = dict()
    val_labels_data = dict()

    for i, image_filepath in enumerate(annotated_image_filepaths):
        # First 16 annotated images go in train labels, last 4 go in val
        if i < 16:
            data = train_labels_data
        else:
            data = val_labels_data

        feature_filepath = image_to_feature_filepath(image_filepath)
        image_id = image_filepath_to_id(image_filepath)
        annotations = all_annotations[image_id]
        data[feature_filepath] = annotations

    classifier_filepath = output_dir / 'classifier1.pkl'
    valresult_filepath = output_dir / 'valresult.json'

    message = TrainClassifierMsg(
        job_token='classifier1',
        trainer_name='minibatch',
        nbr_epochs=10,
        clf_type='MLP',
        train_labels=ImageLabels(data=train_labels_data),
        val_labels=ImageLabels(data=val_labels_data),
        features_loc=DataLocation('filesystem', ''),
        previous_model_locs=[],
        model_loc=DataLocation('filesystem', str(classifier_filepath)),
        valresult_loc=DataLocation('filesystem', str(valresult_filepath)),
    )
    return_message = train_classifier(message)

    ref_accs_str = ", ".join(
        [f"{100*acc:.1f}" for acc in return_message.ref_accs])

    print("------------------------------")
    print(f"Classifier stored at: {classifier_filepath}")
    print(f"New model's accuracy: {100*return_message.acc:.1f}%")
    print(
        "New model's accuracy progression (calculated on part of train_labels)"
        f" after each epoch of training: {ref_accs_str}")

    print(f"Evaluation results:")
    with open(valresult_filepath) as f:
        valresult = json.load(f)
    label_list = [
        label_ids_to_codes[str(label_id)] for label_id in valresult['classes']]
    for ground_truth_i, prediction_i, score in zip(
        valresult['gt'], valresult['est'], valresult['scores']
    ):
        print(f"Actual = {label_list[ground_truth_i]}, Predicted = {label_list[prediction_i]}, Confidence = {100*score:.1f}%")

    print(f"Train time: {return_message.runtime:.1f} s")

    # Classify

    for image_filepath in unannotated_image_filepaths:
        feature_filepath = image_to_feature_filepath(image_filepath)

        message = ClassifyFeaturesMsg(
            job_token=image_filepath.name,
            feature_loc=DataLocation('filesystem', feature_filepath),
            classifier_loc=DataLocation('filesystem', classifier_filepath),
        )
        return_message = classify_features(message)

        print("------------------------------")
        print(f"Classification result for {image_filepath.name}:")

        label_ids = return_message.classes
        for i, (row, col, scores) in enumerate(return_message.scores):
            top_scores = sorted(
                zip(label_ids, scores), key=itemgetter(1), reverse=True)
            top_scores_str = ", ".join([
                f"{label_ids_to_codes[str(label_id)]} = {100*score:.1f}%"
                for label_id, score in top_scores[:TOP_SCORES_PER_POINT]
            ])
            print(f"- Row {row}, column {col}: {top_scores_str}")

        print(f"Classification time: {return_message.runtime:.1f} s")

    print("------------------------------")
    print("Clear the output dir before rerunning this script.")
