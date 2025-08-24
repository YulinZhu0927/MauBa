import numpy as np
import pandas as pd
import airsim
import os

client = airsim.MultirotorClient()
client.confirmConnection()

fig_path = "screen"

pose_df = pd.DataFrame(columns=['index', 'x', 'y', 'z', 'yaw', 'pitch', 'roll'])

num_samples = 50
x_min, x_max, y_min, y_max, z_min, z_max = -4, 4, -4, 4, -5, -2
yaw_min, yaw_max, pitch_min, pitch_max, roll_min, roll_max = -90, 90, -45, 45, -45, 45

camera_list = ["0", '1', '2', '3', '4']

poses_list = []
for i in range(num_samples):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    yaw = np.random.uniform(yaw_min, yaw_max)
    pitch = np.random.uniform(pitch_min, pitch_max)
    roll = np.random.uniform(roll_min, roll_max)

    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))
    poses_list.append({
        'index': i,
        'x': x,
        'y': y,
        'z': z,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll
    })
    client.simSetVehiclePose(pose, True)

    for i, camera in enumerate(camera_list):
        responses = client.simGetImages([airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)])
        img_raw = responses[0]

        img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)

        img_name = f'pose_{i}_x_{x: .2f}_y_{y: .2f}_z_{z: .2f}_yaw_{yaw: .2f}_pitch_{pitch: .2f}_roll_{roll: .2f}_camera_{camera}.png'
        img_path = os.path.join(fig_path, img_name)
        airsim.write_png(os.path.normpath(img_path), img_rgb)
print(f"全部图像和位姿信息均已保存到文件夹：{fig_path}")

pose_df = pd.DataFrame(poses_list)
pose_df.to_csv(os.path.normpath(os.path.join(fig_path, 'pose_df.csv')), index=False)