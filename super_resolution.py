import cv2
import numpy as np
from skimage.transform import resize
from skimage.filters import unsharp_mask, gaussian, sobel, median
from skimage.morphology import disk
from skimage import io
from scipy.ndimage import convolve
import bm3d

def super_resolution_bicubic(image, upscale_factor = 2):
    # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = resized_image
    return sharpened_image

def super_resolution_lanczos(image, upscale_factor = 2):
    # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=4)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = resized_image
    return sharpened_image

def super_resolution_bicubic_gaussian(image, upscale_factor = 2):
    # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = gaussian(resized_image, sigma=1)
    return sharpened_image

def super_resolution_lanczos_gaussian(image, upscale_factor = 2):
    # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=4)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = gaussian(resized_image, sigma=1)
    return sharpened_image

def super_resolution_bicubic_median(image, upscale_factor = 2):
    # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos

    sharpened_image = median(resized_image)
    return sharpened_image

def super_resolution_bicubic_mean(image, upscale_factor = 2):
    # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos
    mean_kernel = np.ones((3, 3)) / 9.0
    sharpened_image = convolve(resized_image, mean_kernel, mode='reflect')
    return sharpened_image

def super_resolution_bicubic_sobel(image, upscale_factor = 2):
   # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = sobel(resized_image)
    return sharpened_image

def super_resolution_bicubic_Unsharp_Mask(image, upscale_factor = 2):
   # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = unsharp_mask(resized_image, radius=1.0, amount=1.0)
    return sharpened_image

def super_resolution_bicubic_laplacian(image, upscale_factor = 2):
   # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=3)  # order=3是Bicubic，可以尝试order=4的Lanczos


    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]], dtype=float)

    # 对灰度图像进行示例。如果是彩色图像，需要对每个通道分别处理。
    if resized_image.ndim == 3:
        # 彩色图像分通道处理
        R = convolve(resized_image[..., 0], laplacian_kernel, mode='reflect')
        G = convolve(resized_image[..., 1], laplacian_kernel, mode='reflect')
        B = convolve(resized_image[..., 2], laplacian_kernel, mode='reflect')
        sharpened_image = np.stack([R, G, B], axis=-1)
    else:
        # 灰度图像处理
        sharpened_image = convolve(resized_image, laplacian_kernel, mode='reflect')
    return sharpened_image

def super_resolution_bicubic_BM3D(image, upscale_factor = 2):
   # 使用Lanczos插值放大（scikit-image的resize支持多种插值方法）
    resized_image = resize(image,
                           (image.shape[0] * upscale_factor, image.shape[1] * upscale_factor),
                           anti_aliasing=True,
                           mode='reflect',
                           order=4)  # order=3是Bicubic，可以尝试order=4的Lanczos

    # 使用Unsharp Mask锐化
    # radius和amount根据需求调整，radius为模糊半径，amount为增强系数
    sharpened_image = bm3d.bm3d(resized_image, sigma_psd=0.05)
    return sharpened_image


# # 读取图像
# img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图像
# # 或者读取彩色图像
# # img = cv2.imread('input_image.jpg')
#
# # 添加椒盐噪声
# noisy_img = add_salt_pepper_noise(img, prob=0.05)
#
# # 显示结果
# cv2.imshow('Original Image', img)
# cv2.imshow('Salt and Pepper Noise', noisy_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 保存结果
# cv2.imwrite('salt_pepper_noise.jpg', noisy_img)

