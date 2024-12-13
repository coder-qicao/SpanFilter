import bm3d
from skimage.filters import median
import numpy as np
from scipy.ndimage import convolve

def denoise_BM3D(img, sigma=0.1):
    """
    Apply BM3D denoising to the input image.
    """
    denoised_image = bm3d.bm3d(img, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return denoised_image


def denoise_median(img):
    denoised_image = median(img)
    return denoised_image


def denoise_mean(img):
    mean_kernel = np.ones((3, 3)) / 9.0
    denoised_image = convolve(img, mean_kernel, mode='reflect')
    return denoised_image