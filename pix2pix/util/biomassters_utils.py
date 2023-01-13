import numpy as np
import tifffile as tif
import warnings


def load_tif(out_path, reshape=False):
    img = tif.imread(out_path)
    if reshape:
        return reshape_tif(img)
    else:
        return img

def reshape_tif(img):
    if len(np.shape(img))==3:
        return np.moveaxis(img,2,0)
    elif len(np.shape(img))==2:
        return img
    else:
        raise ValueError(f"Unknown image shape {np.shape(img)}")

def nanmean(input, axis, catch_warning=False):
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
