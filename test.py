import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def metrics_images(result_image, target_image):
    # 计算MSE
    mse_value = mean_squared_error(target_image, result_image)
    print("MSE:", mse_value)

    # 计算PSNR
    # data_range如果图像是0-1浮点数，可指定为1.0
    psnr_value = peak_signal_noise_ratio(target_image, result_image, data_range=1.0)
    print("PSNR:", psnr_value, "dB")

    # 计算SSIM
    ssim_value = structural_similarity(target_image, result_image, channel_axis=2, data_range=1.0)
    print("SSIM:", ssim_value)

    return mse_value, psnr_value, ssim_value
