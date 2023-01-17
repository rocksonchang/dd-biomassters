import numpy as np
import tifffile as tif
import warnings
import torch


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

def rescale_image(input_image, input_domain=[0, 1], output_domain=[0, 255], clip_input=False):
    """Rescales images

    Parameters:
        input_image (torch.Tensor)     -- input image
        input_domain (List[Int, Int])  -- min and max values of input domain
        output_domain (List[Int, Int]) -- min and max values of output domain

    Returns:
        the rescaled image.

    If input domain is None, rescales using the image max and min
    """
    out_min, out_max = output_domain
    assert out_max > out_min
    if clip_input:
        upper = torch.quantile(input_image, q=.99) #.detach().numpy()
        output_image = torch.clamp(input_image, max=upper)
    else:
        output_image = input_image

    if input_domain:
        in_min, in_max = input_domain
        assert in_max > in_min
        output_image = (output_image - in_min) / (in_max - in_min)
    else:
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_image = output_image * (out_max - out_min) + out_min

    # DEBUG
    # util.summarize_data(input_image, 'rescale input image')
    # util.summarize_data(output_image, 'rescale output image')

    return output_image

def summarize_data(x, label=None):
    print(f'{label}; type: {type(x)}, shape: {x.shape}, max: {x.max():.2f}, min: {x.min():.2f}, mean: {x.mean():.2f}, std: {x.std():.2f}')
