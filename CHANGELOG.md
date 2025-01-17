# Changelog

## 0.5.0

- Generalized feature extractor support by allowing use of any `FeatureExtractor` subclass instance, and extractor files loaded from anywhere (not just from CoralNet's S3 bucket, which requires CoralNet auth).

- In `ExtractFeaturesMsg` and `ClassifyImageMsg`, the parameter `feature_extractor_name` (a string) has been replaced with `extractor` (a `FeatureExtractor` instance).

- In `ExtractFeaturesReturnMsg`, `model_was_cached` has been replaced by `extractor_loaded_remotely`, because now filesystem-caching doesn't apply to some extractor files (they may originally be from the filesystem).

- Config variable `LOCAL_MODEL_PATH` is now `EXTRACTORS_CACHE_DIR`. This is now used by any remote-loaded (S3 or URL based) extractor files. If extractor files are loaded from the filesystem, then it's now possible to run PySpacer without defining any config variable values.

- Added `AWS_REGION` config var, which is now required for S3 usage.

- Added `TEST_EXTRACTORS_BUCKET` and `TEST_BUCKET` config vars for unit tests, but these are not really usable by anyone besides core devs at the moment.

- Some raised errors' types have changed to PySpacer's own `ConfigError` or `HashMismatchError`, and there are cases where error-raising semantics/timing have changed slightly.

## 0.4.1

- Allowed configuration of `MAX_IMAGE_PIXELS`, `MAX_POINTS_PER_IMAGE`, and `MIN_TRAINIMAGES`.

- Previously, if `secrets.json` was present but missing a config value, then pyspacer would go on to look for that config value in Django settings. This is no longer the case; pyspacer now only respects at most one of secrets.json or Django settings (secrets take precedence).

- Updated repo URL from `beijbom/pyspacer` to `coralnet/pyspacer`.

## 0.4.0

- PySpacer now supports Python 3.8+ (testing against 3.8 - 3.10). Support for 3.6 and 3.7 has been dropped.

- Updates to pip-install dependencies:

  - Pillow: >=4.2.0 to >=9.0.1
  - numpy: >=1.17.5 to >=1.19
  - scikit-learn: ==0.22.1 to ==1.1.3 (loading models from older versions should still work)
  - torch: ==1.4.0 to ==1.13.1
  - torchvision: ==0.5.0 to ==0.14.1
  - Removed wget
  - Removed scikit-image
  - Removed tqdm (but it's still a developer requirement)
  - (Removed botocore from dependencies list, but only because it was redundant with boto3 already depending on it)

- Previously, config variables could be specified by a `secrets.json` file or by environment variables. Now a third way is available: a `SPACER` setting in a Django project. Also, the `secrets.json` method no longer uses `SPACER_` prefixes for each variable name. See README for details.

- The `LOCAL_MODELS_PATH` setting is now explicitly required. It was previously not required upfront, but its absence would make some tests fail.

- When an image is sourced from a URL, and the download fails, PySpacer now raises a `SpacerInputError` (instead of a `URLError` for example). The new `SpacerInputError` exception class indicates that the error was most likely caused by the input given to PySpacer (such as an unreachable URL) rather than by a PySpacer bug.

- PySpacer now only configures logging for fire / AWS Batch. When used as a pluggable app, it leaves the existing logging config alone.

## 0.3.1

Upgrade-relevant changes have not been carefully tracked up to this point. If you're unsure how to upgrade, consider starting your environment fresh from 0.4.0 or later.
