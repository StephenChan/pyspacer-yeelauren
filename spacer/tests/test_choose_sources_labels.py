import unittest
from moto import mock_s3
import boto3
from choose_sources_labels import list_folders

class TestListFolders(unittest.TestCase):
    @mock_s3
    def test_list_folders(self):
        # Set up the mock S3
        s3_client = boto3.client('s3', region_name='us-west-2')
        s3_client.create_bucket(Bucket='my-bucket')
        s3_client.put_object(Bucket='my-bucket', Key='my-folder/file1.txt', Body='file1')
        s3_client.put_object(Bucket='my-bucket', Key='my-folder/file2.txt', Body='file2')

        # Call the function
        folders = list_folders(s3_client, 'my-bucket', 'my-folder/')

        # Check the result
        self.assertEqual(folders, ['my-folder/'])

if __name__ == '__main__':
    unittest.main()