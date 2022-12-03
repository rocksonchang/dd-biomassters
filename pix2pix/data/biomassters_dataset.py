import numpy as np
import pandas as pd
import tifffile as tif
from torchvision import transforms

from data.base_dataset import BaseDataset, get_params, get_transform
from util import util


class BioMasstersDataset(BaseDataset):
    """A dataset class for BioMassters

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    RANDOM_STATE = 42
    # Y_SCALE = 63.41566
    # Y_SCALE = 1e4
    Y_SCALE = 385 # 99th percentile

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # prepare metadata
        self.metadata = pd.read_csv(f"{opt.dataroot}/metadata/features_metadata_split_42.csv",index_col=0)
        # self.data = self.metadata[(self.metadata.split == opt.phase) & (self.metadata.is_complete)] \
        #     .chip_id.drop_duplicates().reset_index(drop=True).to_frame()
        self.data = self.metadata[(self.metadata.split == opt.phase) & (self.metadata.is_imputable_s1)] \
            .chip_id.drop_duplicates().reset_index(drop=True).to_frame()
        if opt.max_dataset_size < len(self.data):
            self.data = self.data.sample(opt.max_dataset_size, random_state=self.RANDOM_STATE).reset_index(drop=True)

        # prepare dummy values
        self.dummy_s1_missing_value = np.nan
        self.dummy_s2_missing_value = 255
        self.dummy_s1_missing_img = np.ones([256,256,4])*self.dummy_s1_missing_value
        self.dummy_s2_missing_img = np.ones([256,256,11])*self.dummy_s2_missing_value
        self.s1_missing_value = 9999
        self.s2_missing_value = 255

        # prepare transforms
        self.transform_X = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(
                    (-12.47562, -19.737421, -12.25755, -19.234262) * int(opt.input_nc/4), # S1, month 4 mean
                    (3.3957279, 4.483836, 4.1778703, 5.884509) * int(opt.input_nc/4)      # S1, month 4 std
                )
            ]
        )
        self.transform_y = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0,), (self.Y_SCALE,))         # offset and scale
            ]
        )

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        chip_id = self.data['chip_id'].iloc[index]
        # X_raw = self._load_chip_feature_data(chip_id)
        X_raw = self._load_chip_feature_data_by_quarter(chip_id)
        y_raw = self._load_chip_target_data(chip_id)

        # apply transforms
        X = self.transform_X(X_raw).float()
        y = self.transform_y(y_raw).float()
        
        # DEBUG
        # util.summarize_data(X_raw, 'X raw')
        # util.summarize_data(X, 'X transformed')
        # util.summarize_data(y_raw, 'y raw')
        # util.summarize_data(y, 'y transformed')

        return  {"chip_id": chip_id, "A": X, "B": y, "A_paths": f'A_{chip_id}', "B_paths": f'B_{chip_id}'}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
    
    def _get_chip_metadata(self, chip_id):
        # return self.metadata[(self.metadata.chip_id==chip_id) & (self.metadata.satellite=='S1') & (self.metadata.month==5)]
        # return self.metadata[
        #     (self.metadata.chip_id==chip_id) 
        #     & (self.metadata.satellite=='S1') 
        #     & (self.metadata.month.isin([2,5,8]))
        # ].sort_values(by='month')
        return self.metadata[(self.metadata.chip_id==chip_id) & (self.metadata.satellite=='S1')]
    
    def _load_chip_feature_data(self, chip_id):
        img_channels = []
        for _, row in self._get_chip_metadata(chip_id).iterrows():
            if type(row.filename) != str:
                if row.satellite=='S1':
                    img = self.dummy_s1_missing_img
                elif row.satellite=='S2':
                    img = self.dummy_s2_missing_img
                else:
                    raise ValueError("Unknown satellite value")
            else:
                s3_key = f"{'test' if self.opt.phase=='test' else 'train'}_features/{row.filename}"
                img = self.load_tif(out_path=f'{self.opt.dataroot}/{s3_key}')
                
            img_channels.append(img)
        return np.concatenate(img_channels, axis=2)
    
    def _load_chip_feature_data_by_quarter(self, chip_id):
        metadata = self._get_chip_metadata(chip_id)
        imgs_in_chip = []
        for Q in range(1,5):
            # get images in quarter
            imgs_in_quarter = []
            for _, row in metadata[metadata.Q==Q].iterrows():
                s3_key = f"train_features/{row.filename}"
                _img = self.load_tif(out_path=f'data/{s3_key}')
                # replace encoded missing values with np.nan
                _img[_img==self.s1_missing_value] = np.nan
                imgs_in_quarter.append(_img)
            # take pixel value mean over months in quarter, for each channel
            imgs_in_quarter_by_channel = []
            for ch in range(_img.shape[2]):
                _img = np.nanmean([x[:,:,ch] for x in imgs_in_quarter], axis=0)
                imgs_in_quarter_by_channel.append(_img)
            _img = np.stack(imgs_in_quarter_by_channel, axis=2)
            imgs_in_chip.append(_img)
        return np.concatenate(imgs_in_chip, axis=2)
    
    def _load_chip_target_data(self, chip_id):
        filename = self._get_chip_metadata(chip_id).corresponding_agbm.iloc[0]
        s3_key = f'train_agbm/{filename}'
        img = self.load_tif(out_path=f'{self.opt.dataroot}/{s3_key}')
        return img

    def load_tif(self, out_path, reshape=False):
        img = tif.imread(out_path)
        if reshape:
            return self.reshape_tif(img)
        else:
            return img

    def reshape_tif(self, img):
        if len(np.shape(img))==3:
            return np.moveaxis(img,2,0)
        elif len(np.shape(img))==2:
            return img
        else:
            raise ValueError(f"Unknown image shape {np.shape(img)}")
