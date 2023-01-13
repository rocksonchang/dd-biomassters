import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from data.base_dataset import BaseDataset
from util import biomassters_utils as butils

class BioMasstersDataset(BaseDataset):
    """A dataset class for BioMassters

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    RANDOM_STATE = 42

    # Y_SCALE = 63.41566  # mean
    # Y_SCALE = 1e4       # upper bound
    Y_SCALE = 385         # 99th percentile

    # SATELLITE = 'S1' # S1, S2, None
    CHIP_IS_COMPLETE = False
    CHIP_S1_IS_IMPUTABLE = False

    X_AGGREGATION = 'quarterly'
    # NAN_MEAN_VALUE = -50

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # prepare metadata
        self.metadata = pd.read_csv(opt.metadata_file ,index_col=0)
        condition = self.metadata.split == opt.phase
        if self.CHIP_IS_COMPLETE:
            condition &= self.metadata.is_complete
        if self.CHIP_S1_IS_IMPUTABLE:
            condition &= self.metadata.is_imputable_s1
        self.chips = self.metadata[condition].chip_id.drop_duplicates().reset_index(drop=True).to_frame()

        # sample chips
        self.chips = self.chips.sample(
            min(opt.max_dataset_size, len(self.chips)),
            random_state=self.RANDOM_STATE if not self.opt.random_state else self.opt.random_state
        ).reset_index(drop=True)

        # prepare dummy values
        self.s1_missing_value = -9999
        self.s2_missing_value = 255
        self.dummy_s1_missing_value = np.nan
        self.dummy_s2_missing_value = 255
        self.dummy_s1_missing_img = np.ones([256,256,4])*self.dummy_s1_missing_value
        self.dummy_s2_missing_img = np.concatenate([np.zeros([256,256,10]), np.ones([256,256,1])*100], axis=2) # treat as fully obscured

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
    
    def load(self):
        self.images = []
        for i, row in self.chips.iterrows():
            # print(i, row.chip_id)
            X_raw = self.__load_chip_feature_data(row.chip_id)
            if self.opt.phase == 'test':
                y_raw = None
            else:
                y_raw = self.__load_chip_target_data(row.chip_id)
            self.images.append({'chip_id': row['chip_id'], 'X': X_raw, 'y': y_raw})

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        chip_id = self.chips['chip_id'].iloc[index]
        # if self.X_AGGREGATION == 'quarterly':
        #     X_raw = self.__load_chip_feature_data_by_quarter(chip_id)  # quarterly aggregation
        # else:
        #     X_raw = self.__load_chip_feature_data(chip_id)             # no aggregation
        X_raw = self.__load_chip_feature_data(chip_id)             # no aggregation

        y_raw = self.__load_chip_target_data(chip_id)

        # apply transforms
        X = self.transform_X(X_raw).float()
        X = torch.nan_to_num(X)
        y = self.transform_y(y_raw).float()
        
        # DEBUG
        # util.summarize_data(X_raw, 'X raw')
        # util.summarize_data(X, 'X transformed')
        # util.summarize_data(y_raw, 'y raw')
        # util.summarize_data(y, 'y transformed')

        return  {"chip_id": chip_id, "A": X, "B": y, "A_paths": f'A_{chip_id}', "B_paths": f'B_{chip_id}'}

    def __len__(self):
        """Return the total number of chips in the dataset."""
        return len(self.chips)
    
    def __get_chip_metadata(self, chip_id):
        """ Get chip metadata

        Parameters:
            chip_id (string) - chip id

        Returns:
            A dataframe containing the chip metadata, filtered according to class and runtime arguments.
        """
        condition = self.metadata.chip_id==chip_id
        if self.opt.satellite:
            condition &= self.metadata.satellite==self.opt.satellite
        return self.metadata[condition]

    def __load_chip_feature_data(self, chip_id):
        """ Load chip feature data
        Params:
            chip_id (string) - id of chip to retrieve
        Returns:
            image data (np.Array()) - np array shape (256,256,180) (12 months of S1 4-channel S1 data, 12 month of S2 11-channel data)
        """
        imgs = []
        for _, row in self.__get_chip_metadata(chip_id).iterrows():
            # self.__download_chip_feature_data(row.filename)
            if row.satellite == 'S1':
                img = self.__load_chip_s1_data(row.filename)
            elif row.satellite == 'S2':
                img = self.__load_chip_s2_data(row.filename)
            else:
                raise ValueError("Unknown satellite value")
            imgs.append(img)
        return np.concatenate(imgs, axis=2)

    def __load_chip_s1_data(self, filename):
        if type(filename) != str:
            img = self.dummy_s1_missing_img
        else:
            s3_key = f"{'test' if self.opt.phase=='test' else 'train'}_features/{filename}"
            img = butils.load_tif(out_path=f'{self.opt.dataroot}/{s3_key}')
            img[img==self.s1_missing_value] = self.dummy_s1_missing_value
        return img
    
    def __load_chip_s2_data(self, filename):
        if type(filename) != str:
            img = self.dummy_s2_missing_img
        else:
            s3_key = f"{'test' if self.opt.phase=='test' else 'train'}_features/{filename}"
            img = butils.load_tif(out_path=f'{self.opt.dataroot}/{s3_key}')
            img[:,:,10] = np.clip(img[:,:,10], a_min=0, a_max=100)
        return img

    # def __load_chip_feature_data_by_quarter(self, chip_id):
    #     """Load chip data by quarter

    #     Parameters:
    #         chip_id (string) - chip id

    #     Returns:
    #         List[np.Array()] -- list length 4, np array shape (256,256,4) for S1 and (256,256,11) for s2

    #     Pixel values are averaged over each month in a quarter and for every channel. Nans are ignored.
    #     """
    #     metadata = self.__get_chip_metadata(chip_id)
    #     imgs_in_chip = []
    #     for Q in range(1,5):
    #         # get images in quarter
    #         imgs_in_quarter = []
    #         for _, row in metadata[metadata.Q==Q].iterrows():
    #             s3_key = f"train_features/{row.filename}"
    #             _img = butils.load_tif(out_path=f'data/{s3_key}')
    #             # replace encoded missing values with np.nan
    #             _img[_img==self.s1_missing_value] = np.nan
    #             imgs_in_quarter.append(_img)
    #         # take pixel value mean over months in quarter, for each channel
    #         imgs_in_quarter_by_channel = []
    #         for ch in range(_img.shape[2]):
    #             _img = butils.nanmean([x[:,:,ch] for x in imgs_in_quarter], axis=0, catch_warning=True)
    #             imgs_in_quarter_by_channel.append(_img)
    #         _img = np.stack(imgs_in_quarter_by_channel, axis=2)
    #         imgs_in_chip.append(_img)
    #     return imgs_in_chip

    def __load_chip_target_data(self, chip_id):
        filename = self.__get_chip_metadata(chip_id).corresponding_agbm.iloc[0]
        s3_key = f'train_agbm/{filename}'
        img = butils.load_tif(out_path=f'{self.opt.dataroot}/{s3_key}')
        return img
