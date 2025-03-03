import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os


# 计算重合度的函数
def calculate_overlap(image1, image2):
    """
    计算两个二值化图像中像素重合度（同时为 255 的像素）。

    参数:
    image1, image2: 二值化图像 (NumPy 数组)，值应为 0 或 255，形状相同

    返回:
    overlap_count: 重合像素数量（同时为 255 的像素数）
    overlap_ratio: 重合比例（重合像素数除以总像素数）
    overlap_mask: 重合区域的布尔掩码
    """
    # 确保图像形状相同
    if image1.shape != image2.shape:
        raise ValueError("两个图像的形状必须相同")
    print("我是尺寸！",image1.shape)
    # 强制二值化当前图像（image2）
    image2 = (image2 > 0).astype(np.uint8) * 255

    # 调试：打印图像值范围
    # print(f"image1 unique values: {np.unique(image1)}")
    # print(f"image2 unique values: {np.unique(image2)}")

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
    print("我是！", overlap_ratio)
    return overlap_count, overlap_ratio, overlap


# 自定义环境类
class SquareAlignmentEnv(gym.Env):
    def __init__(self, template_image_path, image_list):
        super(SquareAlignmentEnv, self).__init__()

        # 读取并处理模板图像（灰度模式，二值化）
        self.template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        if self.template_image is None:
            raise ValueError("无法读取模板图片")

        # 缩小模板图像尺寸（例如 224x224）
        target_size = (1280, 1280)  # 目标尺寸
        self.template_image = cv2.resize(self.template_image, target_size, interpolation=cv2.INTER_NEAREST)

        # 确保模板图像是严格二值化的（0 和 255）
        _, self.template_binary = cv2.threshold(self.template_image, 127, 255, cv2.THRESH_BINARY)
        self.template_binary = (self.template_binary > 0).astype(np.uint8) * 255  # 强制二值化
        self.height, self.width = self.template_binary.shape  # 使用缩小后的高度和宽度

        # 计算模板白色区域的中心和尺寸
        white_pixels = np.where(self.template_binary == 255)
        if len(white_pixels[0]) == 0:
            raise ValueError("模板图片中没有白色区域（255）")
        self.template_center = (
            int(np.mean(white_pixels[1])),  # x 中心（列）
            int(np.mean(white_pixels[0]))   # y 中心（行）
        )
        self.template_size = int(np.max(white_pixels[1]) - np.min(white_pixels[1]))  # 宽度作为尺寸

        self.image_list = image_list

        # 基于数据调整动作空间：平移（-7到7 像素）和旋转（-5.65到8.0 度）
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),  # 平移 x, 平移 y, 旋转
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 调整 observation_space 为缩小后的图像格式（灰度图像，[height, width, 1]）
        self.observation_space = spaces.Box(
            low=0,  # 最小值为 0（黑色）
            high=255,  # 最大值为 255（白色）
            shape=(self.height, self.width, 1),  # 灰度图像，通道数为 1
            dtype=np.uint8
        )

        self.max_steps = 100
        self.current_image = None
        self.current_binary = None
        self.rotation_angle = 0  # 初始旋转角度（度）
        self.square_x = 0  # 初始 x 位置
        self.square_y = 0  # 初始 y 位置
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_image_path = np.random.choice(self.image_list)
        self.current_image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        if self.current_image is None:
            raise ValueError(f"无法读取图片: {self.current_image_path}")

        # 缩小当前图像尺寸（例如 224x224）
        target_size = (self.width, self.height)  # 使用与模板相同的尺寸
        self.current_image = cv2.resize(self.current_image, target_size, interpolation=cv2.INTER_NEAREST)

        # 确保当前图像是严格二值化的（0 和 255）
        _, self.current_binary = cv2.threshold(self.current_image, 127, 255, cv2.THRESH_BINARY)
        self.current_binary = (self.current_binary > 0).astype(np.uint8) * 255  # 强制二值化

        # 验证二值化结果（调试用）
        if not np.all(np.isin(self.current_binary, [0, 255])):
            print(f"警告：二值化后仍包含非 0/255 值：{np.unique(self.current_binary)}")
            self.current_binary = (self.current_binary > 0).astype(np.uint8) * 255  # 再次强制

        # 调试：打印图像值范围
        # print(f"Original image unique values: {np.unique(self.current_image)}")
        # print(f"Binary image unique values: {np.unique(self.current_binary)}")

        # 计算当前图像白色区域的初始中心
        white_pixels = np.where(self.current_binary == 255)
        if len(white_pixels[0]) == 0:
            print(f"警告：{self.current_image_path} 中没有白色区域，使用随机初始值")
            self.square_x = int(np.random.randint(0, self.width))
            self.square_y = int(np.random.randint(0, self.height))
            self.rotation_angle = np.random.uniform(-5.65, 8.0)  # 初始随机旋转角度（度）
        else:
            self.square_x = int(np.mean(white_pixels[1]))  # x 中心（列）
            self.square_y = int(np.mean(white_pixels[0]))  # y 中心（行）
            self.rotation_angle = 0  # 初始无旋转（度）

        self.current_step = 0

        # 返回当前二值化图像作为观察（调整为 [height, width, 1] 格式）
        observation = np.expand_dims(self.current_binary, axis=-1)  # 增加通道维度，形状为 [height, width, 1]
        return observation, {}

    def step(self, action):
        # 动作：[-1, 1] -> 平移和旋转（基于数据调整步长）
        # 平移步长：最大平移量 ±7 像素，分 10 步（动作值 [-1, 1] 映射到 ±7）
        move_step = 7.0  # 最大平移量（像素）
        # 旋转步长：最大旋转 ±3.35 度（约 -3.35 到 2.7 度），分 10 步（更细粒度控制）
        rotation_step = 0.5  # 调整为 0.5 度

        delta_x = action[0] * move_step  # 水平移动（-7 到 7 像素）
        delta_y = action[1] * move_step  # 垂直移动（-7 到 7 像素）
        delta_rotation = action[2] * rotation_step  # 旋转角度变化（-0.5 到 0.5 度）

        self.square_x += delta_x
        self.square_y += delta_y
        self.rotation_angle += delta_rotation

        # 限制在图片范围内，并转换为整数
        self.square_x = int(np.clip(self.square_x, 0, self.width))
        self.square_y = int(np.clip(self.square_y, 0, self.height))
        self.rotation_angle = np.clip(self.rotation_angle, -5.65, 8.0)  # 限制旋转角度（度）

        self.current_step += 1

        # 移动并旋转当前图像并用白色填充
        current_binary = self._move_rotate_and_fill(self.current_binary, self.square_x, self.square_y, self.rotation_angle)

        state = np.expand_dims(current_binary, axis=-1)  # 调整为 [height, width, 1] 格式
        reward = float(self._get_reward())
        terminated = bool(self._is_done())
        truncated = bool(self.current_step >= self.max_steps)
        info = {}
        print(f"Rotation angle: {self.rotation_angle} degrees, Delta rotation: {delta_rotation} degrees")
        print(f'平移的是x:{self.square_x},平移的是y:{self.square_y}')
        return state, reward, terminated, truncated, info

    def _get_state(self):
        # 不再需要单独的 _get_state 方法，因为观察直接从 step 和 reset 返回
        pass

    def _get_reward(self):
        # 移动并旋转当前图像并用白色填充
        current_binary = self._move_rotate_and_fill(self.current_binary, self.square_x, self.square_y, self.rotation_angle)

        # 计算重合度作为奖励
        overlap_count, overlap_ratio, _ = calculate_overlap(self.template_binary, current_binary)
        reward = overlap_ratio  # 直接使用重合比例作为奖励

        # 惩罚过多的步数
        reward -= 0.01 * self.current_step / self.max_steps

        return float(reward)

    def _move_rotate_and_fill(self, image, center_x, center_y, angle):
        """
        移动和旋转图像的中心位置，保持尺寸不变，用白色（255）填充超出边界部分。

        参数:
        image: 输入二值化图像（灰度，0 或 255）
        center_x, center_y: 新的中心位置
        angle: 旋转角度（度）

        返回:
        二值化后的移动和旋转图像（0 和 255）
        """
        # 获取图像尺寸
        height, width = image.shape

        # 创建空白画布（白色填充）
        canvas = np.ones((self.height, self.width), dtype=np.uint8) * 255

        # 确保 center_x 和 center_y 是整数
        center_x = int(center_x)
        center_y = int(center_y)

        # 计算偏移量
        offset_x = center_x - width // 2
        offset_y = center_y - height // 2

        # 确保切片索引是整数
        x_start = int(max(0, -offset_x))
        y_start = int(max(0, -offset_y))
        x_end = int(min(width, self.width - offset_x))
        y_end = int(min(height, self.height - offset_y))

        canvas_x_start = int(max(0, offset_x))
        canvas_y_start = int(max(0, offset_y))
        canvas_x_end = int(min(self.width, offset_x + width))
        canvas_y_end = int(min(self.height, offset_y + height))

        # 临时画布用于旋转（与 image 形状相同）
        temp_canvas = np.ones_like(image, dtype=np.uint8) * 255  # 使用 image 的形状初始化
        if y_end > y_start and x_end > x_start:
            temp_canvas[y_start:y_end, x_start:x_end] = image[y_start:y_end, x_start:x_end]

        # 旋转图像（angle 已经是度，直接使用）
        center = (width // 2, height // 2)  # 旋转中心
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 直接使用度
        rotated = cv2.warpAffine(temp_canvas, rotation_matrix, (width, height), borderValue=255)

        # 将旋转后的图像粘贴到画布上
        if canvas_y_end > canvas_y_start and canvas_x_end > canvas_x_start:
            canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = rotated[
                :canvas_y_end - canvas_y_start, :canvas_x_end - canvas_x_start
            ]

        return canvas

    def _is_done(self):
        # 移动并旋转当前图像并用白色填充
        current_binary = self._move_rotate_and_fill(self.current_binary, self.square_x, self.square_y, self.rotation_angle)

        # 计算重合度
        _, overlap_ratio, _ = calculate_overlap(self.template_binary, current_binary)
        return overlap_ratio > 0.94  # 当重合比例超过 0.9 时结束

    def render(self, mode='human'):
        # 移动并旋转当前图像并用白色填充
        current_binary = self._move_rotate_and_fill(self.current_binary, self.square_x, self.square_y, self.rotation_angle)

        # 创建彩色结果图像
        result_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        overlap, _, _ = calculate_overlap(self.template_binary, current_binary)

        # 重合处标记为绿色，其他处为灰色（模板为蓝色，当前为红色）
        result_image[overlap] = [0, 255, 0]  # 重合处为绿色
        result_image[self.template_binary == 255] = [255, 0, 0]  # 模板为红色
        result_image[current_binary == 255] = [0, 0, 255]  # 当前为蓝色

        # 保存图像到文件夹（创建输出目录）
        output_dir = r"D:\Project\HYJ\RL\RL_F2\output"  # 更新输出路径
        os.makedirs(output_dir, exist_ok=True)
        step_img_path = os.path.join(output_dir, f"step_{self.current_step}.png")
        cv2.imwrite(step_img_path, result_image)
        print('图片已保存')

        # 显示图像
        cv2.imshow("Alignment", result_image)
        cv2.waitKey(50)


# 设置图片路径
template_image_path = r"D:\Project\HYJ\RL\RL_F2\pic\std\standard_639.png"
image_dir = r"D:\Project\HYJ\RL\RL_F2\pic\250303"
image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png'))]

# 创建并检查环境
env = SquareAlignmentEnv(template_image_path, image_list)
check_env(env)

# 训练 PPO 模型，使用 CnnPolicy
# model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, device='cuda')  # 显式使用 GPU
# 或者，如果遇到 GPU 效率问题，可以用：
model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, n_steps=256, device='cuda')  # 使用 CPU，减少 n_steps
model.learn(total_timesteps=200000)

# 测试模型
print('running_test')
obs, _ = env.reset()
env.render()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        print("Alignment achieved!")
        break

cv2.destroyAllWindows()