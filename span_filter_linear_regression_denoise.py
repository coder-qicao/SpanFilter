import numpy as np
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
from sklearn.linear_model import LinearRegression
from denoise import denoise_median, denoise_BM3D, denoise_mean
from test import metrics_images
import os

# 假设训练集中有多张彩色图像（格式可为jpg、png等）
# 训练图像（LR和对应的GT（HR）图像路径）
noisy_image_path = "dataset/noise_img/baby.png"
clear_image_path = "dataset/Set5/LRbicx3/baby.png"  # Ground Truth 高分辨率图像，用于监督信号
# filters
F1 = denoise_BM3D
F2 = denoise_median

# 读取低分辨率图像
lr_img = io.imread(noisy_image_path)
lr_img = img_as_float(lr_img)
if lr_img.ndim == 2:
    lr_img = np.stack([lr_img, lr_img, lr_img], axis=-1)
elif lr_img.shape[2] != 3:
    raise ValueError("Training image is not 3-channel RGB.")

# 读取高分辨率目标图像
hr_img = io.imread(clear_image_path)
hr_img = img_as_float(hr_img)
if hr_img.ndim == 2:
    hr_img = np.stack([hr_img, hr_img, hr_img], axis=-1)
elif hr_img.shape[2] != 3:
    raise ValueError("HR target image is not 3-channel RGB.")

# 从LR图像提取特征
R = lr_img[..., 0]
G = lr_img[..., 1]
B = lr_img[..., 2]

F1_R = F1(R)
F2_R = F2(R)

F1_G = F1(G)
F2_G = F2(G)

F1_B = F1(B)
F2_B = F2(B)

# 将特征和目标打包成X与Y
# 对每个通道独立建立线性回归模型：Y_channel = w1 * F1_channel + w2 * F2_channel + bias
H, W = F1_R.shape

F1_R_flat = F1_R.flatten()
F2_R_flat = F2_R.flatten()
F1_G_flat = F1_G.flatten()
F2_G_flat = F2_G.flatten()
F1_B_flat = F1_B.flatten()
F2_B_flat = F2_B.flatten()

R_target = hr_img[...,0].flatten()
G_target = hr_img[...,1].flatten()
B_target = hr_img[...,2].flatten()

# 对R通道回归
X_R = np.column_stack((F1_R_flat, F2_R_flat))
lr_R = LinearRegression()
lr_R.fit(X_R, R_target)
w1_R, w2_R = lr_R.coef_
b_R = lr_R.intercept_

# 对G通道回归
X_G = np.column_stack((F1_G_flat, F2_G_flat))
lr_G = LinearRegression()
lr_G.fit(X_G, G_target)
w1_G, w2_G = lr_G.coef_
b_G = lr_G.intercept_

# 对B通道回归
X_B = np.column_stack((F1_B_flat, F2_B_flat))
lr_B = LinearRegression()
lr_B.fit(X_B, B_target)
w1_B, w2_B = lr_B.coef_
b_B = lr_B.intercept_

print("R channel weights:", w1_R, w2_R, "bias:", b_R)
print("G channel weights:", w1_G, w2_G, "bias:", b_G)
print("B channel weights:", w1_B, w2_B, "bias:", b_B)

# 对新图像进行处理（这里使用同一张图作为新图像，也可换其他LR图像）

# 定义要进行测试的图像列表
input_folder_path = "dataset/noise14/"
input_files = os.listdir(input_folder_path)
gt_folder_path = "dataset/Set14/Set14/image_SRF_3/LR/"
gt_input_files = os.listdir(gt_folder_path)
test_images = list(zip(input_files, gt_input_files))


# 用于存储所有图像的评估指标
psnr_results = []
ssim_results = []
mse_results = []
psnr_results_f1 = []
ssim_results_f1 = []
mse_results_f1 = []
psnr_results_f2 = []
ssim_results_f2 = []
mse_results_f2 = []

for (new_image_path, new_target_image_path) in test_images:
    new_image = io.imread(input_folder_path+new_image_path)
    new_image = img_as_float(new_image)

    new_target_image = io.imread(gt_folder_path+new_target_image_path)
    new_target_image = img_as_float(new_target_image)

    if new_image.ndim == 2:
        new_image = np.stack([new_image, new_image, new_image], axis=-1)
    if new_target_image.ndim == 2:
        new_target_image = np.stack([new_target_image, new_target_image, new_target_image], axis=-1)

    R_new = new_image[..., 0]
    G_new = new_image[..., 1]
    B_new = new_image[..., 2]

    F1_R_new = F1(R_new)
    F2_R_new = F2(R_new)
    F1_G_new = F1(G_new)
    F2_G_new = F2(G_new)
    F1_B_new = F1(B_new)
    F2_B_new = F2(B_new)

    # 利用学到的回归权重对新图像进行预测
    R_result = w1_R * F1_R_new + w2_R * F2_R_new + b_R
    G_result = w1_G * F1_G_new + w2_G * F2_G_new + b_G
    B_result = w1_B * F1_B_new + w2_B * F2_B_new + b_B

    result_image = np.stack([R_result, G_result, B_result], axis=-1)
    result_image = np.clip(result_image, 0, 1)
    result_image_u8 = img_as_ubyte(result_image)

    img_name = new_image_path.split('/')[-1].split('.')[0]
    io.imsave(f'results/denoise_LR/result_{img_name}_lr_color.png', result_image_u8)
    print(f"Result saved as results/result_{img_name}_lr_color.png")

    # 输出F1和F2加权结果
    result_image_f1 = np.stack([F1_R_new, F1_G_new, F1_B_new], axis=-1)
    result_image_f1 = np.clip(result_image_f1, 0, 1)
    io.imsave(f'results/denoise_LR/result_{img_name}_f1_color.png', img_as_ubyte(result_image_f1))

    result_image_f2 = np.stack([F2_R_new, F2_G_new, F2_B_new], axis=-1)
    result_image_f2 = np.clip(result_image_f2, 0, 1)
    io.imsave(f'results/denoise_LR/result_{img_name}_f2_color.png', img_as_ubyte(result_image_f2))

    # 对结果进行评估（假设 metrics_images 返回 (psnr, ssim)）
    mse, psnr, ssim = metrics_images(result_image, new_target_image)
    mse_f1, psnr_f1, ssim_f1 = metrics_images(result_image_f1, new_target_image)
    mse_f2, psnr_f2, ssim_f2 = metrics_images(result_image_f2, new_target_image)

    psnr_results.append(psnr)
    ssim_results.append(ssim)
    mse_results.append(mse)
    psnr_results_f1.append(psnr_f1)
    ssim_results_f1.append(ssim_f1)
    mse_results_f1.append(mse_f1)
    psnr_results_f2.append(psnr_f2)
    ssim_results_f2.append(ssim_f2)
    mse_results_f2.append(mse_f2)

# 所有测试图像处理完后计算平均值
print("Average PSNR (Span Filter):", np.mean(psnr_results))
print("Average SSIM (Span Filter):", np.mean(ssim_results))
print("Average MSE (Span Filter):", np.mean(mse_results))

print("Average PSNR (F1):", np.mean(psnr_results_f1))
print("Average SSIM (F1):", np.mean(ssim_results_f1))
print("Average MSE (F1):", np.mean(mse_results_f1))

print("Average PSNR (F2):", np.mean(psnr_results_f2))
print("Average SSIM (F2):", np.mean(ssim_results_f2))
print("Average MSE (F2):", np.mean(mse_results_f2))

