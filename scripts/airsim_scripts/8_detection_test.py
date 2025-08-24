import airsim
import cv2
import numpy as np
import time

# 连接到 AirSim 模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 设置摄像头名称和图像类型为深度图（DepthPerspective）
camera_name = "0"
image_type = airsim.ImageType.DepthPerspective

while True:
    # 获取深度图像（返回压缩后的图像字节流）
    raw_depth = client.simGetImage(camera_name, image_type)
    if raw_depth is None:
        print("No depth image received.")
        time.sleep(0.1)
        continue

    # 将字节流转换为 numpy 数组，并使用 cv2.IMREAD_UNCHANGED 保留原始深度数据
    np_img = np.frombuffer(raw_depth, dtype=np.uint8)
    depth_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print("Failed to decode depth image.")
        continue

    # 输出深度图原始形状和数据类型，便于调试
    print("Original depth image shape:", depth_img.shape, depth_img.dtype)

    # 深度图可能是 16 位或 32 位单通道图，归一化到 0-255 范围，并转换为 8 位图像
    depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)

    # 如果图像有多余的通道（例如 (H, W, 1)），则将其压缩为 (H, W)
    if len(depth_normalized.shape) == 3 and depth_normalized.shape[2] == 1:
        depth_normalized = np.squeeze(depth_normalized, axis=2)

    # 输出归一化图像形状，确认其为单通道 8 位图像
    print("Normalized depth image shape:", depth_normalized.shape, depth_normalized.dtype)

    # 应用彩色映射
    try:
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    except cv2.error as e:
        print("Error applying colormap:", e)
        continue

    cv2.imshow("AirSim Depth Image", depth_colored)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
