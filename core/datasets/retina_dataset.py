"""Retina dataset."""

import re
import tensorflow_datasets as tfds

_DESCRIPTION = """\
Retinal OCT image dataset reflecting Drusen, DME, CNV and Normal 
"""

_CITATION = """\
title = {Retinal OCT image data}
author = {paultimothymooney}
publisher = {Kaggle}
url = {https://www.kaggle.com/paultimothymooney/kermany2018 }
"""
_TRAIN_URL = "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_train.zip"
_TEST_URL = "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_test.zip"
_LABELS = ["NORMAL", "DRUSEN", "DME", "CNV"]

_NAME_RE = re.compile(r"^([\w]*[\\/])(NORMAL|DRUSEN|DME|CNV)(?:/|\\)[\w-]*\.jpeg$")

class RetinaDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for retinaoct dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names=_LABELS)
        }),
        supervised_keys=("image", "label"),
        homepage='https://www.kaggle.com/paultimothymooney/kermany2018',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    train_path, test_path = dl_manager.download([_TRAIN_URL, _TEST_URL])

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "archive": dl_manager.iter_archive(train_path)
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "archive": dl_manager.iter_archive(test_path)
            }),
  
    ]

  def _generate_examples(self, archive):
    """Generate images and labels given the directory path.

    Args:
      archive: object that iterates over the zip.

    Yields:
      The image path and its corresponding label.
    """

    for fname, fobj in archive:
      res = _NAME_RE.match(fname)
      if not res:
        continue
      label = res.group(2)
      record = {
          "image": fobj,
          "label": label,
      }
      yield fname, record

