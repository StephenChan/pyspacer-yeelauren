import unittest

import numpy as np
from PIL import Image

from spacer import config
from spacer.data_classes import ImageFeatures
from spacer.extract_features import DummyExtractor, FeatureExtractor
from spacer.messages import ExtractFeaturesReturnMsg, DataLocation
from spacer.storage import load_image, storage_factory
from .common import TEST_EXTRACTORS
from .decorators import \
    require_caffe, require_test_extractors, require_test_fixtures


class TestDummyExtractor(unittest.TestCase):

    def test_simple(self):

        extractor = DummyExtractor(
            feature_dim=4096,
        )
        features, return_msg = extractor(
            im=Image.new('RGB', (100, 100)),
            rowcols=[(100, 100)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 100)
        self.assertEqual(features.point_features[0].col, 100)

    def test_dims(self):

        feature_dim = 42
        extractor = DummyExtractor(
            feature_dim=feature_dim,
        )
        self.assertEqual(extractor.feature_dim, feature_dim)

    def test_duplicate_rowcols(self):

        extractor = DummyExtractor(
            feature_dim=4096,
        )
        rowcols = [(100, 100), (100, 100), (50, 50)]
        features, return_msg = extractor(
            im=Image.new('RGB', (100, 100)),
            rowcols=rowcols,
        )

        self.assertEqual(
            len(features.point_features), len(rowcols),
            msg="Duplicate rowcols should be preserved, not merged")


@require_caffe
@require_test_extractors
@require_test_fixtures
class TestCaffeExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = FeatureExtractor.deserialize(TEST_EXTRACTORS['vgg16'])

    def setUp(self):
        config.filter_warnings()

    def test_simple(self):

        img = load_image(DataLocation(
            storage_type='s3',
            key='edinburgh3.jpg',
            bucket_name=config.TEST_BUCKET,
        ))
        features, return_msg = self.extractor(
            im=img,
            rowcols=[(100, 100)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 100)
        self.assertEqual(features.point_features[0].col, 100)

    def test_dims(self):
        self.assertEqual(self.extractor.feature_dim, 4096)

    def test_corner_case1(self):
        """
        This particular image caused trouble on the coralnet production server.
        The image file itself is lightly corrupted. Pillow can only read it
        if LOAD_TRUNCATED_IMAGES is set to True.
        """
        img = load_image(DataLocation(
            storage_type='s3',
            key='kh6dydiix0.jpeg',
            bucket_name=config.TEST_BUCKET,
        ))
        features, return_msg = self.extractor(
            im=img,
            rowcols=[(148, 50), (60, 425)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 148)
        self.assertEqual(features.point_features[0].col, 50)

    def test_corner_case2(self):
        """
        Another corrupted image seen in coralnet production.
        """
        img = load_image(DataLocation(
            storage_type='s3',
            key='sfq2mr5qbs.jpeg',
            bucket_name=config.TEST_BUCKET,
        ))
        features, return_msg = self.extractor(
            im=img,
            rowcols=[(190, 226), (25, 359)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 190)
        self.assertEqual(features.point_features[0].col, 226)

    def test_regression(self):
        """
        This test runs the extractor on a known image and compares the
        results to the features extracted with
        https://github.com/beijbom/ecs_spacer/releases/tag/1.0
        """
        rowcols = [(20, 265),
                   (76, 295),
                   (59, 274),
                   (151, 62),
                   (265, 234)]

        img = load_image(DataLocation(
            storage_type='s3',
            key='08bfc10v7t.png',
            bucket_name=config.TEST_BUCKET,
        ))
        features_new, _ = self.extractor(
            im=img,
            rowcols=rowcols,
        )

        legacy_feat_loc = DataLocation(storage_type='s3',
                                       key='08bfc10v7t.png.featurevector',
                                       bucket_name=config.TEST_BUCKET)
        features_legacy = ImageFeatures.load(legacy_feat_loc)

        for pf_new, pf_legacy in zip(features_new.point_features,
                                     features_legacy.point_features):
            self.assertTrue(np.allclose(pf_legacy.data, pf_new.data,
                                        atol=1e-5))
            self.assertTrue(pf_legacy.row is None)
            self.assertTrue(pf_new.row is not None)


@require_test_extractors
@require_test_fixtures
class TestEfficientNetExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = FeatureExtractor.deserialize(
            TEST_EXTRACTORS['efficientnet-b0'])

    def setUp(self):
        config.filter_warnings()

    def test_simple(self):

        img = load_image(DataLocation(
            storage_type='s3',
            key='edinburgh3.jpg',
            bucket_name=config.TEST_BUCKET,
        ))
        features, return_msg = self.extractor(
            im=img,
            rowcols=[(100, 100)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 100)
        self.assertEqual(features.point_features[0].col, 100)

        self.assertEqual(len(features.point_features[0].data), 1280)
        self.assertEqual(features.feature_dim, 1280)

    def test_dims(self):
        self.assertEqual(self.extractor.feature_dim, 1280)

    def test_corner_case1(self):

        img = load_image(DataLocation(
            storage_type='s3',
            key='kh6dydiix0.jpeg',
            bucket_name=config.TEST_BUCKET,
        ))
        features, return_msg = self.extractor(
            im=img,
            rowcols=[(148, 50), (60, 425)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 148)
        self.assertEqual(features.point_features[0].col, 50)
        self.assertEqual(len(features.point_features[0].data), 1280)

    def test_corner_case2(self):

        img = load_image(DataLocation(
            storage_type='s3',
            key='sfq2mr5qbs.jpeg',
            bucket_name=config.TEST_BUCKET,
        ))
        features, return_msg = self.extractor(
            im=img,
            rowcols=[(190, 226), (25, 359)],
        )

        self.assertTrue(isinstance(return_msg, ExtractFeaturesReturnMsg))
        self.assertTrue(isinstance(features, ImageFeatures))

        # Check some feature metadata
        self.assertEqual(features.point_features[0].row, 190)
        self.assertEqual(features.point_features[0].col, 226)
        self.assertEqual(len(features.point_features[0].data), 1280)

    def test_regression(self):
        rowcols = [(20, 265),
                   (76, 295),
                   (59, 274),
                   (151, 62),
                   (265, 234)]

        img = load_image(DataLocation(
            storage_type='s3',
            key='08bfc10v7t.png',
            bucket_name=config.TEST_BUCKET,
        ))
        features_new, _ = self.extractor(
            im=img,
            rowcols=rowcols,
        )

        legacy_feat_loc = DataLocation(storage_type='s3',
                                       key='08bfc10v7t.png.effnet.'
                                           'ver1.featurevector',
                                       bucket_name=config.TEST_BUCKET)
        features_legacy = ImageFeatures.load(legacy_feat_loc)

        self.assertFalse(features_legacy.valid_rowcol)
        self.assertEqual(features_legacy.npoints, len(rowcols))
        self.assertEqual(features_legacy.feature_dim, 1280)

        self.assertTrue(features_new.valid_rowcol)
        self.assertEqual(features_new.npoints, len(rowcols))
        self.assertEqual(features_new.feature_dim, 1280)

        for pf_new, pf_legacy in zip(features_new.point_features,
                                     features_legacy.point_features):
            self.assertTrue(np.allclose(pf_legacy.data, pf_new.data,
                                        atol=1e-5))
            self.assertTrue(pf_legacy.row is None)
            self.assertTrue(pf_new.row is not None)


@require_test_extractors
class TestExtractorCache(unittest.TestCase):

    def test(self):
        extractor = FeatureExtractor.deserialize(TEST_EXTRACTORS['vgg16'])
        file_storage = storage_factory('filesystem')
        key = 'definition'

        # Empty the cache.
        filepath_for_cache = str(extractor.data_filepath_for_cache(key))
        if file_storage.exists(filepath_for_cache):
            file_storage.delete(filepath_for_cache)

        # Test cache miss.
        filepath_loaded, remote_loaded = \
            extractor.load_data_into_filesystem(key)
        self.assertTrue(remote_loaded)
        self.assertTrue(file_storage.exists(filepath_loaded))
        self.assertEqual(filepath_for_cache, filepath_loaded)

        # Test cache hit.
        _, remote_loaded = extractor.load_data_into_filesystem(key)
        self.assertFalse(remote_loaded)


if __name__ == '__main__':
    unittest.main()
