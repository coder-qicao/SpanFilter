import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
import glob
from denoise import denoise_BM3D, denoise_median
from test import metrics_images
import os

# 假设训练集中有多张彩色图像（格式可为jpg、png等）
# 训练图像（LR和对应的GT（HR）图像路径）
noisy_image_path = "dataset/noise5/baby.png"
clear_image_path = "dataset/Set5/LRbicx2/baby.png"  # Ground Truth 高分辨率图像，用于监督信号
# filters
F1 = denoise_BM3D
F2 = denoise_median

data_points = []

# 对每张图像进行处理
img = io.imread(noisy_image_path)
img = img_as_float(img)  # 转换为float类型，范围[0,1]

# 确保是三通道彩色图像
if img.ndim == 2:
    # 如果是灰度图，转换成RGB格式
    img = np.stack([img, img, img], axis=-1)

# 拆分RGB通道
R = img[..., 0]
G = img[..., 1]
B = img[..., 2]

# 对每个通道卷积
F1_R = F1(R)
F2_R = F2(R)

F1_G = F1(G)
F2_G = F2(G)

F1_B = F1(B)
F2_B = F2(B)

# 将特征展开为N_pixels x 6 的矩阵
# 每个像素点给出(F1_R, F2_R, F1_G, F2_G, F1_B, F2_B)特征
F1_R_flat = F1_R.flatten()
F2_R_flat = F2_R.flatten()
F1_G_flat = F1_G.flatten()
F2_G_flat = F2_G.flatten()
F1_B_flat = F1_B.flatten()
F2_B_flat = F2_B.flatten()

pixel_features = np.column_stack((F1_R_flat, F2_R_flat,
                                  F1_G_flat, F2_G_flat,
                                  F1_B_flat, F2_B_flat))
data_points.append(pixel_features)

# 合并所有图像的数据
data_points = np.vstack(data_points)  # shape: (N * H * W, 6)

# 对数据进行PCA
pca = PCA(n_components=1)  # 寻找主方向
pca.fit(data_points)

# 主成分方向
principal_direction = pca.components_[0]  # shape: (6,)
print("Principal direction (weights):", principal_direction)
w1_R, w2_R, w1_G, w2_G, w1_B, w2_B = principal_direction

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
    R_result = w1_R * F1_R_new + w2_R * F2_R_new
    G_result = w1_G * F1_G_new + w2_G * F2_G_new
    B_result = w1_B * F1_B_new + w2_B * F2_B_new

    result_image = np.stack([R_result, G_result, B_result], axis=-1)
    result_image = np.clip(result_image, 0, 1)
    result_image_u8 = img_as_ubyte(result_image)

    img_name = new_image_path.split('/')[-1].split('.')[0]
    io.imsave(f'results/denoise_PCA/result_{img_name}_lr_color.png', result_image_u8)
    print(f"Result saved as results/result_{img_name}_lr_color.png")

    # 输出F1和F2加权结果
    result_image_f1 = np.stack([F1_R_new, F1_G_new, F1_B_new], axis=-1)
    result_image_f1 = np.clip(result_image_f1, 0, 1)
    io.imsave(f'results/denoise_PCA/result_{img_name}_f1_color.png', img_as_ubyte(result_image_f1))

    result_image_f2 = np.stack([F2_R_new, F2_G_new, F2_B_new], axis=-1)
    result_image_f2 = np.clip(result_image_f2, 0, 1)
    io.imsave(f'results/denoise_PCA/result_{img_name}_f2_color.png', img_as_ubyte(result_image_f2))

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

