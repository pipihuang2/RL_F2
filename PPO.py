import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os


# 优化后的检测正方形函数
def detect_square(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h if h != 0 else 0
        if 0.8 <= aspect_ratio <= 1.2:
            return (x, y, w, h)
    return None


# 自定义环境类
class SquareAlignmentEnv(gym.Env):
    def __init__(self, template_image_path, image_list):
        super(SquareAlignmentEnv, self).__init__()

        self.template_image = cv2.imread(template_image_path)
        self.image_list = image_list
        self.image_size = self.template_image.shape[0]
        template_box = detect_square(self.template_image)
        if template_box is None:
            raise ValueError("无法在模板图片中检测到正方形")
        self.template_x, self.template_y, self.template_size, _ = template_box
        self.template_center = (self.template_x + self.template_size // 2,
                                self.template_y + self.template_size // 2)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0.9], dtype=np.float32),
            high=np.array([self.image_size, self.image_size, 1.15], dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = 100
        self.current_image = None
        self.current_box = None
        self.max_distance = np.sqrt(2) * self.image_size
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_image_path = np.random.choice(self.image_list)
        self.current_image = cv2.imread(self.current_image_path)
        self.current_box = detect_square(self.current_image)
        if self.current_box is None:
            print(f"警告：无法检测 {self.current_image_path} 中的正方形，使用随机初始值")
            self.square_x = np.random.randint(0, self.image_size)
            self.square_y = np.random.randint(0, self.image_size)
            self.scale = np.random.uniform(0.9, 1.15)
        else:
            x, y, w, _ = self.current_box
            self.square_x = x + w // 2
            self.square_y = y + w // 2
            self.scale = max(0.9, min(1.15, w / self.template_size))

        self.current_step = 0
        return self._get_state(), {}

    def step(self, action):
        # 计算当前正方形中心与模板中心的距离
        current_center = np.array([self.square_x, self.square_y])
        template_center = np.array(self.template_center)
        distance = np.sqrt(np.sum((current_center - template_center) ** 2))

        # 动态调整平移步长
        if distance > 50:  # 距离远（>50像素），用较大步长
            move_step = 5
        elif distance < 20:  # 距离近（<20像素），用较小步长
            move_step = 1
        else:  # 中间距离（20-50像素），线性插值
            move_step = 1 + (distance - 20) * (5 - 1) / (50 - 20)  # 线性从1到5

        # 动作：[-1, 1] -> 动态平移和固定缩放
        delta_x = action[0] * move_step  # 动态步长平移
        delta_y = action[1] * move_step  # 动态步长平移
        delta_scale = action[2] * 0.02  # 保持缩放步长

        self.square_x += delta_x
        self.square_y += delta_y
        self.scale += delta_scale

        # 限制在图片和缩放范围内
        self.square_x = np.clip(self.square_x, 0, self.image_size)
        self.square_y = np.clip(self.square_y, 0, self.image_size)
        self.scale = np.clip(self.scale, 0.8, 1.2)  # 保持缩放范围 [0.8, 1.2]

        self.current_step += 1

        state = self._get_state()
        reward = float(self._get_reward())
        terminated = bool(self._is_done())
        truncated = bool(self.current_step >= self.max_steps)
        info = {}
        return state, reward, terminated, truncated, info

    def _get_state(self):
        return np.array([self.square_x, self.square_y, self.scale], dtype=np.float32)

    def _get_reward(self):
        # 计算 IoU
        template_left = self.template_center[0] - self.template_size // 2
        template_right = self.template_center[0] + self.template_size // 2
        template_top = self.template_center[1] - self.template_size // 2
        template_bottom = self.template_center[1] + self.template_size // 2

        scaled_size = int(self.template_size * self.scale)
        current_left = self.square_x - scaled_size // 2
        current_right = self.square_x + scaled_size // 2
        current_top = self.square_y - scaled_size // 2
        current_bottom = self.square_y + scaled_size // 2

        inter_left = max(template_left, current_left)
        inter_right = min(template_right, current_right)
        inter_top = max(template_top, current_top)
        inter_bottom = min(template_bottom, current_bottom)

        inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)

        template_area = self.template_size * self.template_size
        current_area = scaled_size * scaled_size
        union_area = template_area + current_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        # 优化奖励：IoU = 0 时给小负奖励，低 IoU 线性增加
        if iou == 0:
            reward = -0.1  # 小负奖励，鼓励探索
        elif iou < 0.5:
            reward = 0.5 * iou  # 低 IoU 线性奖励
        else:
            reward = iou  # 高 IoU 使用原始奖励
        if iou > 0.9:  # IoU > 0.9 额外奖励
            reward += 0.5

        return float(reward)

    # def _get_reward(self):
    #     distance = np.sqrt((self.square_x - self.template_center[0]) ** 2 +
    #                        (self.square_y - self.template_center[1]) ** 2)
    #     scale_diff = abs(self.scale - 1.0)
    #     distance_reward = -distance / self.max_distance
    #     scale_reward = -scale_diff / 1.5
    #     reward = distance_reward + scale_reward
    #     if distance < 10 and scale_diff < 0.05:  # 缩小阈值
    #         reward += 1.0
    #     return float(reward)

    # def _is_done(self):
    #     distance = np.sqrt((self.square_x - self.template_center[0]) ** 2 +
    #                        (self.square_y - self.template_center[1]) ** 2)
    #     scale_diff = abs(self.scale - 1.0)
    #     return distance < 10 and scale_diff < 0.05  # 缩小阈值

    def _is_done(self):
        # 计算 IoU
        template_left = self.template_center[0] - self.template_size // 2
        template_right = self.template_center[0] + self.template_size // 2
        template_top = self.template_center[1] - self.template_size // 2
        template_bottom = self.template_center[1] + self.template_size // 2

        scaled_size = int(self.template_size * self.scale)
        current_left = self.square_x - scaled_size // 2
        current_right = self.square_x + scaled_size // 2
        current_top = self.square_y - scaled_size // 2
        current_bottom = self.square_y + scaled_size // 2

        inter_left = max(template_left, current_left)
        inter_right = min(template_right, current_right)
        inter_top = max(template_top, current_top)
        inter_bottom = min(template_bottom, current_bottom)

        inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)

        template_area = self.template_size * self.template_size
        current_area = scaled_size * scaled_size
        union_area = template_area + current_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        # 结束条件：IoU > 0.95（几乎完全重合）
        return iou > 0.9

    def render(self, mode='human'):
        img = self.current_image.copy()
        scaled_size = int(self.template_size * self.scale)
        x_offset = int(self.square_x - scaled_size // 2)
        y_offset = int(self.square_y - scaled_size // 2)

        cv2.rectangle(img, (x_offset, y_offset),
                      (x_offset + scaled_size, y_offset + scaled_size),
                      (0, 255, 0), 2)

        cv2.rectangle(img,
                      (self.template_center[0] - self.template_size // 2,
                       self.template_center[1] - self.template_size // 2),
                      (self.template_center[0] + self.template_size // 2,
                       self.template_center[1] + self.template_size // 2),
                      (0, 0, 255), 2)

        # 保存图像到文件夹（创建输出目录）
        output_dir = r"F:\deeplearning\pytorch\Reinforcement_Learning\one_e\output"
        os.makedirs(output_dir, exist_ok=True)
        step_img_path = os.path.join(output_dir, f"step_{self.current_step}.png")
        cv2.imwrite(step_img_path, img)
        print('图片已保存')

        # 显示图像
        cv2.imshow("Alignment", img)
        cv2.waitKey(50)


# 设置图片路径
template_image_path = r"F:\deeplearning\pytorch\Reinforcement_Learning\one_e\template\f108_120_245.4_1.00.png"
image_dir = r"F:\deeplearning\pytorch\Reinforcement_Learning\one_e\pic"
image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# 创建并检查环境
env = SquareAlignmentEnv(template_image_path, image_list)
check_env(env)

# 训练PPO模型
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
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