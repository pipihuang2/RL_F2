import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_image_with_square(image_size=200, square_size=50):
    # 创建空白图像
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    # 模板位置（固定在中央）
    template_x, template_y = image_size // 2, image_size // 2

    # 随机生成正方形的位置、旋转角度和缩放比例
    square_x = np.random.randint(50, image_size - square_size)
    square_y = np.random.randint(50, image_size - square_size)
    angle = np.random.uniform(0, 360)
    # scale = np.random.uniform(0.9, 1.2)

    scale = 1

    # 创建正方形
    square = np.ones((square_size, square_size), dtype=np.uint8) * 255
    scaled_size = int(square_size * scale)
    square = cv2.resize(square, (0, 0),fx=scale,fy=scale)

    # 旋转正方形
    M = cv2.getRotationMatrix2D((scaled_size // 2, scaled_size // 2), angle, 1)
    # square_rotated = cv2.warpAffine(square, M, (scaled_size, scaled_size))

    # 将正方形放入图像
    h, w = square.shape
    x_offset = square_x - w // 2
    y_offset = square_y - h // 2
    if x_offset >= 0 and y_offset >= 0 and x_offset + w <= image_size and y_offset + h <= image_size:
        image[100: 100 + h, 100:100 + w] = square

    return image, (square_x, square_y, angle, scale)


# 生成并显示示例图像
for i in range(1):
    image, params = generate_image_with_square()
    # plt.imshow(image, cmap='gray')
    plt.imsave(f"template/f{params[0]}_{params[1]}_{params[2]:.1f}_{params[3]:.2f}.png",image,cmap='gray')
    # plt.title(f"x: {params[0]}, y: {params[1]}, angle: {params[2]:.1f}, scale: {params[3]:.2f}")
    # plt.show()

