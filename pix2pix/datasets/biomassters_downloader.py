import argparse
from botocore.exceptions import ClientError
import boto3
import os
import numpy as np
import pandas as pd
import tifffile as tif


class BioMasstersDownloader():
    S3_BUCKET_NAME = 'drivendata-competition-biomassters-public-us'
    RANDOM_STATE = 42

    def __init__(self, phase, dataroot, dataset_size, metadata_file, satellite, chip_is_complete, is_imputable_s1):
        
        self.phase = phase
        self.dataroot = dataroot
        self.dataset_size = dataset_size

        # prepare metadata
        self.metadata = pd.read_csv(metadata_file,index_col=0)
        # self.metadata = self.metadata[~self.metadata.filename.isna()]
        if chip_is_complete:
            self.metadata = self.metadata[self.metadata.is_complete]
        elif is_imputable_s1:
            self.metadata = self.metadata[self.metadata.is_imputable_s1]
        if satellite:
            self.metadata = self.metadata[self.metadata.satellite==satellite]

        # unique chips
        self.data = self.metadata[self.metadata.split == self.phase] \
            .chip_id.drop_duplicates().reset_index(drop=True).to_frame()
        if self.dataset_size < len(self.data):
            self.data = self.data.sample(self.dataset_size, random_state=self.RANDOM_STATE).reset_index(drop=True)

        self.images = []
        self.__initialize_s3()

        # prepare dummy values
        self.dummy_s1_missing_value = 256
        self.dummy_s2_missing_value = -9999        
        self.dummy_s1_missing_img = np.ones([256,256,4])*self.dummy_s1_missing_value
        self.dummy_s2_missing_img = np.ones([256,256,11])*self.dummy_s2_missing_value
    
    def __initialize_s3(self):
        """Initalize AWS s3 bucket for 
        """
        s3_resource = boto3.resource(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
        self.s3_bucket = s3_resource.Bucket(self.S3_BUCKET_NAME)

    def run(self, load_tif=False):
        for i, row in self.data.iterrows():
            print(i, row.chip_id)
            X = self.__load_chip_feature_data(row['chip_id'], load_tif)
            y = self.__load_chip_target_data(row['chip_id'], load_tif)
            if load_tif:
                self.images.append({'chip_id': row['chip_id'], 'X': X, 'y': y})

    def __get_chip_metadata(self, chip_id):
        return self.metadata[self.metadata.chip_id==chip_id]
            
    def __load_chip_feature_data(self, chip_id, load_tif):
        img_channels = []
        for _, row in self.__get_chip_metadata(chip_id).iterrows():
            if type(row.filename) != str:
                if load_tif:
                    if row.satellite=='S1':
                        img = self.dummy_s1_missing_img
                    elif row.satellite=='S2':
                        img = self.dummy_s2_missing_img
                    else:
                        raise ValueError("Unknown satellite value")
                    img_channels.append(img)
            else:
                s3_key = f"{'test' if self.phase=='test' else 'train'}_features/{row.filename}"
                img = self.__download_obj_from_s3(self.s3_bucket, s3_key, out_path=f'{self.dataroot}/{s3_key}')
                if load_tif:
                    img = self.__load_tif(out_path=f'{self.dataroot}/{s3_key}')
                    img_channels.append(img)
        if load_tif:
            return np.concatenate(img_channels, axis=2)

    def __load_chip_target_data(self, chip_id, load_tif):
        filename = self.__get_chip_metadata(chip_id).corresponding_agbm.iloc[0]
        s3_key = f'train_agbm/{filename}'
        img = self.__download_obj_from_s3(self.s3_bucket, s3_key, out_path=f'{self.dataroot}/{s3_key}')
        if load_tif:
            img = self.__load_tif(out_path=f'{self.dataroot}/{s3_key}')
            return img
    
    def __load_tif(self, out_path):
        img = tif.imread(out_path)
        return img

    def __download_obj_from_s3(self, s3_bucket, key, out_path):
        if not os.path.exists(out_path):
            try:
                s3_bucket.download_file(key, out_path)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                else:
                    raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--dataroot', type=str, default='data', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--dataset_size', type=int, default=5, help='Number of chips to load.')
    parser.add_argument('--metadata_file', type=str, default='data', help='path to metadata csv')
    parser.add_argument('--satellite', type=str, default=None, help='Satellite, S1 or S2 (default None)')
    parser.add_argument('--chip_is_complete', action='store_true', help='if specified, only download complete chips')
    parser.add_argument('--is_imputable_s1', action='store_true', help='if specified, only download imputable s1 chips')
    opt, _ = parser.parse_known_args()

    metadata_file = f"data/metadata/features_metadata_split_42.csv"
    dd = BioMasstersDownloader(
        phase=opt.phase,
        dataroot=opt.dataroot,
        dataset_size=opt.dataset_size,
        metadata_file=opt.metadata_file,
        satellite=opt.satellite,
        chip_is_complete=opt.chip_is_complete,
        is_imputable_s1=opt.is_imputable_s1
    )
    dd.run(load_tif=False)