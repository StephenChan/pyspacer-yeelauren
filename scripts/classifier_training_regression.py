import glob
import json
import os

import boto
import fire
import tqdm

import warnings

from spacer import config
from spacer.data_classes import ImageLabels
from spacer.storage import storage_factory
from spacer.train_classifier import trainer_factory


class ClassifierRegressionTest:
    """
    This runs a training on exported features from CoralNet.
    It assumes read permission on the "spacer-trainingdata" bucket.

    All data is formatted per the management command in
    ...
    """

    # TODO: add permalink to management command --^

    @staticmethod
    def _cache_local(source_root, image_root, export_name, source_id):

        """ Download source data to local """
        conn = boto.connect_s3()
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)
        if not os.path.exists(source_root):
            os.mkdir(source_root)
        if not os.path.exists(image_root):
            os.mkdir(image_root)

        mdkey = bucket.get_key('{}/s{}/meta.json'.format(
            export_name,
            source_id))
        mdkey.get_contents_to_filename(
            os.path.join(source_root, 'meta.json'))

        img_keys = bucket.list(prefix='{}/s{}/images'.format(
            export_name,
            source_id))

        img_keys = [key for key in img_keys if key.name.endswith('json')]

        print("-> Downloading {} metadata and feature files...".
              format(len(img_keys)))
        for key in tqdm.tqdm(img_keys):
            _, filename = key.name.split('images')
            local_path = os.path.join(image_root, filename.lstrip('/'))
            if not os.path.exists(local_path):
                key.get_contents_to_filename(local_path)

    def train(self,
              source_id: int,
              local_path: str,
              n_epochs: int = 5,
              export_name: str = 'beta_export_v2'):

        # Sci-kit learns calibration step throws out a ton of warnings.
        # That we don't need to see here.
        warnings.simplefilter('ignore', RuntimeWarning)

        source_root = os.path.join(local_path, 's{}'.format(source_id))
        image_root = os.path.join(source_root, 'images')

        # Download all data to local.
        # Train and eval will run much faster that way...
        print('Downloading data for source id: {}.'.format(source_id))
        self._cache_local(source_root, image_root, export_name, source_id)

        # Create the train and val ImageLabels data structures.
        print('-> Assembling train and val data for source id: {}'.format(
            source_id))
        files = glob.glob(os.path.join(image_root, "*.json"))
        train_labels = ImageLabels(data={})
        val_labels = ImageLabels(data={})
        for itt, filename in enumerate(files):
            if 'anns' in filename:
                with open(filename) as fp:
                    anns = json.load(fp)
                meta_filename = filename.replace('anns', 'meta')
                with open(meta_filename) as fp:
                    meta = json.load(fp)
                if meta['in_trainset']:
                    labels = train_labels
                else:
                    assert meta['in_valset']
                    labels = val_labels
                labels.data[filename.replace('anns', 'features')] = [
                    (ann['row'], ann['col'], ann['label']) for ann in anns
                ]

        # Store and compile the TrainClassifierMsg
        storage = storage_factory('filesystem', '')
        traindata_key = os.path.join(source_root, 'traindata.json')
        valdata_key = os.path.join(source_root, 'valdata.json')
        storage.store_string(
            traindata_key,
            json.dumps(train_labels.serialize()))
        storage.store_string(
            valdata_key,
            json.dumps(val_labels.serialize()))

        # Perform training
        print("-> Training...")
        trainer = trainer_factory('minibatch')
        clf, val_results, return_message = trainer(
            traindata_key, valdata_key, n_epochs, [], storage)
        with open(os.path.join(source_root, 'meta.json')) as fp:
            source_meta = json.load(fp)

        print('Re-trained {} ({}). Old acc: {:.1f}, new acc: {:.1f}'.format(
            source_meta['name'],
            source_meta['pk'],
            100 * float(source_meta['best_robot_accuracy']),
            100 * return_message.acc)
        )

    @staticmethod
    def list(export_name: str = 'beta_export_v2'):
        """ Lists sources available in export. """

        conn = boto.connect_s3(config.AWS_ACCESS_KEY_ID,
                               config.AWS_SECRET_ACCESS_KEY)
        bucket = conn.get_bucket('spacer-trainingdata', validate=True)

        source_keys = bucket.list(prefix='{}/s'.format(export_name),
                                  delimiter='images')
        meta_keys = [key for key in source_keys if key.name.endswith('json')]

        header_format = '{:>30}, {:>4}, {:>6}, {}\n{}'
        print(header_format.format('Name', 'id', 'n_imgs', 'acc (%)', '-'*53))
        entry_format = '{:>30}, {:>4}, {:>6}, {:.1f}%'
        for meta_key in meta_keys:
            md = json.loads(meta_key.get_contents_as_string().decode('UTF-8'))
            print(entry_format.format(
                md['name'][:20],
                md['pk'],
                md['nbr_confirmed_images'],
                100*float(md['best_robot_accuracy'])))


if __name__ == '__main__':
    fire.Fire(ClassifierRegressionTest)