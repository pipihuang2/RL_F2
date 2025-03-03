import numpy as np
import cv2


def calculate_overlap(image1, image2):
    """
    计算两个二值化图像中像素重合度（同时为 255 的像素）。

    参数:
    image1, image2: 二值化图像 (NumPy 数组)，值应为 0 或 255，形状相同

    返回:
    overlap_count: 重合像素数量（同时为 255 的像素数）
    overlap_ratio: 重合比例（重合像素数除以总像素数）
    """
    # 确保图像形状相同
    if image1.shape != image2.shape:
        raise ValueError("两个图像的形状必须相同")

    # 确保是二值图像（只包含 0 和 255）
    if not (np.all(np.isin(image1, [0, 255])) and np.all(np.isin(image2, [0, 255]))):
        raise ValueError("图像必须是二值化图像（只包含 0 和 255）")

    # 找到同时为 255 的像素（逻辑与操作）
    overlap = np.logical_and(image1 == 255, image2 == 255)

    # 计算重合像素的数量
    overlap_count = np.sum(overlap)

    # 计算总像素数
    total_pixels = image1.size

    # 计算重合比例（相对于总像素数）
    overlap_ratio = overlap_count / total_pixels if total_pixels > 0 else 0

    return overlap_count, overlap_ratio, overlap


# 示例使用
if __name__ == "__main__":
    # 创建示例二值化图像（4x4）
    image1 = cv2.imread("D:\Project\HYJ\RL\RL_F2\pic\std\standard_639.png")
    image2  = cv2.imread(r"D:\Project\HYJ\RL\RL_F2\pic\250303\01.png")
    # 计算重合度
    overlap_count, overlap_ratio, overlap = calculate_overlap(image1, image2)

    print(f"重合像素数量: {overlap_count}")
    print(f"重合比例（相对于总像素数）: {overlap_ratio:.4f}")

    # # 可视化重合区域（可选）
    # overlap_image = np.zeros_like(image1, dtype=np.uint8)
    # overlap_image[overlap] = 255  # 重合处标记为 255
    # cv2.imwrite('out.jpg',overlap_image)
    # # cv2.imshow("Overlap", overlap_image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()