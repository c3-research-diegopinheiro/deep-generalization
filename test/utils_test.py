import os
import shutil
import unittest
from utils.consts import dataset_structure
from utils.mkdir_dataset import mkdir_dataset


class UtilsTest(unittest.TestCase):

    def test_should_generate_dataset_initial_structure(self):
        dataset_test_name = 'dataset_test_name'

        mkdir_dataset(dataset_test_name)

        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/train'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/test'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/validation'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/train/yes'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/train/no'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/test/yes'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/test/no'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/validation/yes'), True)
        self.assertEqual(os.path.exists('DATASET/' + dataset_test_name + '/validation/no'), True)

        try:
            shutil.rmtree('DATASET/' + dataset_test_name)
        except OSError as e:
            print(e)
