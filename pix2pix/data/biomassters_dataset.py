import copy
import numpy as np
import pandas as pd
import tifffile as tif
import torch
from torchvision import transforms
import warnings

from data.base_dataset import BaseDataset, get_params, get_transform
from util import util


class BioMasstersDataset(BaseDataset):
    """A dataset class for BioMassters

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    RANDOM_STATE = 42
    METADATA_FILE = "./data/metadata/features_metadata_split_42.csv"

    # Y_SCALE = 63.41566
    # Y_SCALE = 1e4
    Y_SCALE = 385 # 99th percentile

    SATELLITE = 'S1' # S1, S2, None
    CHIP_IS_COMPLETE = False
    CHIP_S1_IS_IMPUTABLE = False

    X_AGGREGATION = 'quarterly'
    NAN_MEAN_VALUE = -50

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # prepare metadata
        self.metadata = pd.read_csv(self.METADATA_FILE ,index_col=0)
        condition = self.metadata.split == opt.phase
        if self.CHIP_IS_COMPLETE:
            condition &= self.metadata.is_complete
        if self.CHIP_S1_IS_IMPUTABLE:
            condition &= self.metadata.is_imputable_s1
        self.data = self.metadata[condition].chip_id.drop_duplicates().reset_index(drop=True).to_frame()

        if opt.max_dataset_size < len(self.data):
            self.data = self.data.sample(opt.max_dataset_size, random_state=self.RANDOM_STATE).reset_index(drop=True)

        # prepare dummy values
        self.s1_missing_value = -9999
        self.s2_missing_value = 255
        self.dummy_s1_missing_value = np.nan
        self.dummy_s2_missing_value = 255
        self.dummy_s1_missing_img = np.ones([256,256,4])*self.dummy_s1_missing_value
        self.dummy_s2_missing_img = np.ones([256,256,11])*self.dummy_s2_missing_value

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
        if self.X_AGGREGATION == 'quarterly':
            X_raw = self.__load_chip_feature_data_by_quarter(chip_id)  # quarterly aggregation
        else:
            X_raw = self.__load_chip_feature_data(chip_id)             # no aggregation
        
        # # impute chip values
        # if np.isnan(X_raw).any():
        #     print('chip_id', chip_id, 'imputation: mean_annual_value_per_channel')
        #     X_raw = self.impute_chip_values(X_raw, strategy='mean_annual_value_per_channel')
        # if np.isnan(X_raw).any():
        #     print('chip_id', chip_id, 'imputation: mean_chip_value_per_channel')
        #     X_raw = self.impute_chip_values(X_raw, strategy='mean_chip_value_per_channel')
        # if np.isnan(X_raw).any():
        #     print('chip_id', chip_id, 'imputation: mean_chip_value')
        #     X_raw = self.impute_chip_values(X_raw, strategy='mean_chip_value')

        X_raw = np.concatenate(X_raw, axis=2)
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
        """Return the total number of images in the dataset."""
        return len(self.data)
    
    def __get_chip_metadata(self, chip_id):
        """ Get chip metadata

        Parameters:
            chip_id (string) - chip id

        Returns:
            A dataframe containing the chip metadata, filtered according to class and runtime arguments.
        """
        condition = self.metadata.chip_id==chip_id
        if self.SATELLITE:
            condition &= self.metadata.satellite==self.SATELLITE
        return self.metadata[condition]
    
    def __load_chip_feature_data(self, chip_id):
        """ Load chip feature data
        Params:
            chip_id (string) - id of chip to retrieve
        Returns:
            image data (List[np.Array()]) - list length 12, np array shape (256,256,4) for S1 and (256,256,11) for s2
        """
        img_channels = []
        for _, row in self.__get_chip_metadata(chip_id).iterrows():
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
        return img_channels
    
    def __load_chip_feature_data_by_quarter(self, chip_id):
        """Load chip data by quarter

        Parameters:
            chip_id (string) - chip id

        Returns:
            List[np.Array()] -- list length 4, np array shape (256,256,4) for S1 and (256,256,11) for s2

        Pixel values are averaged over each month in a quarter and for every channel. Nans are ignored.
        """
        metadata = self.__get_chip_metadata(chip_id)
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
                _img = self.nanmean([x[:,:,ch] for x in imgs_in_quarter], axis=0, catch_warning=True)
                imgs_in_quarter_by_channel.append(_img)
            _img = np.stack(imgs_in_quarter_by_channel, axis=2)
            imgs_in_chip.append(_img)
        return imgs_in_chip

    def impute_chip_values(self, input, strategy):
        """ Impute chip values
        Parameters:
            imgs (List[np.Array()]) - list length 4 (quarterly agg) or 12 (monthly), np array shape (256,256,4) for S1 and (256,256,11) for s2
            strategy (str) - imputation strategy {fixed_value, mean_annual_value_per_channel, mean_chip_value_per_channel, mean_chip_value}
        
        Returns:
            imputed images (List[np.Array()])
        """
        imgs = copy.deepcopy(input)
        num_channels = 4
        if strategy == 'fixed_value':
            for idx in range(len(imgs)):
                imgs[idx] = np.nan_to_num(imgs[idx], nan=self.NAN_MEAN_VALUE)
        elif strategy == 'mean_annual_value_per_channel':
            for ch in range(num_channels):
                img_channel = [x[:,:,ch] for x in imgs]
                mean = self.nanmean(img_channel, axis=0, catch_warning=True)
                for idx in range(len(imgs)):
                    isnan_idx = np.isnan(imgs[idx][:,:,ch])
                    imgs[idx][:,:,ch][isnan_idx] = mean[isnan_idx]
        elif strategy == 'mean_chip_value_per_channel':
            for ch in range(num_channels):
                img_channel = [x[:,:,ch] for x in imgs]
                mean = self.nanmean(img_channel, axis=None, catch_warning=True)
                for idx in range(len(imgs)):
                    isnan_idx = np.isnan(imgs[idx][:,:,ch])
                    imgs[idx][:,:,ch][isnan_idx] = mean
        elif strategy == 'mean_chip_value':
            mean = self.nanmean(imgs, axis=None, catch_warning=False)
            for idx in range(len(imgs)):
                imgs[idx] = np.nan_to_num(imgs[idx], nan=mean)
        else:
            raise Warning(f'Imputation strategy {strategy} not recognized')
        return imgs

    def __load_chip_target_data(self, chip_id):
        filename = self.__get_chip_metadata(chip_id).corresponding_agbm.iloc[0]
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

    def nanmean(self, input, axis, catch_warning=False):
        """ Numpy nanmean, wrapped by catch warning
        Parameters:
            input (np.array) - input array (or list of array)
            axis (int) - axis for aggregaion
            catch_warning (bool) - if true, catch and suppress warning
        Returns:
        """
        with warnings.catch_warnings():
            # nanmean across images in quarter may produce RuntimeWarning
            if catch_warning:
                warnings.simplefilter("ignore")
            return np.nanmean(input, axis=axis)
