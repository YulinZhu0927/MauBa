# import json
# import random
#
# from tqdm import tqdm
#
# # 定义任务类别及每类任务数量
# categories = ["Composite Instruction", "Path Planning", "Obstacle Avoidance", "Target Search"]
# tasks_per_category = 100
#
# # 定义难度分布比例（简单:中等:复杂），此处可根据研究数据调整
# difficulty_distribution = {"Simple": 0.5, "Medium": 0.3, "Complex": 0.2}
#
# # 动作描述词库（中文），用于丰富任务描述的措辞
# actions = {
#     "takeoff": ["起飞", "垂直起飞", "升空"],
#     "land": ["降落", "着陆"],
#     "move_forward": ["向前飞行", "前进"],
#     "move_backward": ["向后飞行", "后退"],
#     "move_left": ["向左飞行", "左移"],
#     "move_right": ["向右飞行", "右移"],
#     "ascend": ["上升", "爬升"],
#     "descend": ["下降", "降低高度"],
#     "hover": ["悬停", "停留在空中"],
#     "rotate": ["旋转", "转向"],
#     "avoid": ["避开障碍", "绕过障碍"],
#     "search": ["搜索", "寻找"],
#     "take_photo": ["拍照", "捕获图像"]
# }
#
# # 随机选择动作描述短语的辅助函数
# def phrase(action_key):
#     return random.choice(actions[action_key])
#
# # 生成“复合指令”任务（由2~3个基础API指令组合而成）
# def generate_composite_task(difficulty):
#     code_lines = []
#     desc_actions = []  # 用于组合描述文本的动作列表
#
#     # 初始化无人机（连接AirSim并起飞）
#     code_lines.append("import airsim")
#     code_lines.append("client = airsim.MultirotorClient()")
#     code_lines.append("client.confirmConnection()")
#     code_lines.append("client.enableApiControl(True)")
#     code_lines.append("client.armDisarm(True)")
#     code_lines.append("client.takeoffAsync().join()")
#
#     # 根据难度确定基础动作数量：简单1~2个，中等2~3个，复杂2~3个（含更丰富动作）
#     if difficulty == "Simple":
#         num_core_actions = 1 if random.random() < 0.5 else 2
#         action_pool = ["move"]  # 简单任务主要包含移动
#     elif difficulty == "Medium":
#         num_core_actions = 2 if random.random() < 0.7 else 3
#         action_pool = ["move", "rotate", "hover"]  # 中等任务包含移动和旋转、悬停
#     else:  # Complex
#         num_core_actions = 2 if random.random() < 0.4 else 3
#         action_pool = ["move", "rotate", "hover", "take_photo"]  # 复杂任务可能包括拍照等
#
#     # 逐个生成基础动作指令
#     for _ in range(num_core_actions):
#         action_type = random.choice(action_pool)
#         if action_type == "move":
#             # 随机选择移动方向并生成相应代码
#             direction = random.choice(["forward", "back", "left", "right", "ascend", "descend"])
#             if direction in ["forward", "back"]:
#                 distance = random.randint(5, 20)  # 前后移动距离（米）
#                 speed = 5.0  # 设定移动速度 (m/s)
#                 vx = speed if direction == "forward" else -speed
#                 duration = distance / speed
#                 code_lines.append(f"client.moveByVelocityAsync({vx}, 0, 0, {duration}).join()")
#                 desc_actions.append(f"{phrase('move_forward')}{distance}米" if direction == "forward"
#                                      else f"{phrase('move_backward')}{distance}米")
#             elif direction in ["left", "right"]:
#                 distance = random.randint(5, 20)  # 左右移动距离
#                 speed = 5.0
#                 vy = speed if direction == "right" else -speed
#                 duration = distance / speed
#                 code_lines.append(f"client.moveByVelocityAsync(0, {vy}, 0, {duration}).join()")
#                 desc_actions.append(f"{phrase('move_right')}{distance}米" if direction == "right"
#                                      else f"{phrase('move_left')}{distance}米")
#             elif direction == "ascend":
#                 altitude = random.randint(2, 10)  # 上升高度（米）
#                 code_lines.append(f"client.moveByVelocityAsync(0, 0, -2, {altitude/2}).join()")
#                 desc_actions.append(f"{phrase('ascend')}{altitude}米")
#             elif direction == "descend":
#                 altitude = random.randint(2, 10)  # 下降高度（米）
#                 code_lines.append(f"client.moveByVelocityAsync(0, 0, 2, {altitude/2}).join()")
#                 desc_actions.append(f"{phrase('descend')}{altitude}米")
#         elif action_type == "rotate":
#             # 随机选择旋转角度并生成代码（正为顺时针，负为逆时针）
#             angle = random.choice([45, 90, 180, -45, -90])
#             code_lines.append(f"client.rotateToYawAsync({angle}).join()")
#             if angle >= 0:
#                 desc_actions.append(f"{phrase('rotate')}向右{angle}度")
#             else:
#                 desc_actions.append(f"{phrase('rotate')}向左{abs(angle)}度")
#         elif action_type == "hover":
#             # 悬停若干秒
#             hover_time = random.randint(2, 5)
#             code_lines.append("client.hoverAsync().join()")
#             code_lines.append(f"time.sleep({hover_time})  # 悬停{hover_time}秒")
#             desc_actions.append(f"{phrase('hover')}{hover_time}秒")
#         elif action_type == "take_photo":
#             # 拍照动作（使用AirSim相机API获取图像）
#             code_lines.append("responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])")
#             code_lines.append("# 在此处可以对获取的图像进行处理或保存")
#             desc_actions.append(phrase('take_photo'))
#
#     # 任务结束后降落，并释放控制
#     code_lines.append("client.hoverAsync().join()")
#     code_lines.append("time.sleep(2)")
#     code_lines.append("client.landAsync().join()")
#     code_lines.append("client.armDisarm(False)")
#     code_lines.append("client.enableApiControl(False)")
#
#     # 生成任务描述文本（包含主要动作步骤）
#     # 将动作短语拼接成连贯的中文描述句子
#     if desc_actions:
#         # 多个动作时，用“然后”连接动作描述
#         description = "无人机起飞后，" + "，然后".join(desc_actions) + "，最后降落。"
#     else:
#         description = "无人机起飞后直接降落。"
#     return {"task_type": "复合指令", "difficulty": difficulty, "description": description, "code": "\n".join(code_lines)}
#
# # 生成“路径规划”任务
# def generate_path_planning_task(difficulty):
#     code_lines = []
#     # 初始化无人机并起飞
#     code_lines.extend([
#         "import airsim",
#         "client = airsim.MultirotorClient()",
#         "client.confirmConnection()",
#         "client.enableApiControl(True)",
#         "client.armDisarm(True)",
#         "client.takeoffAsync().join()"
#     ])
#
#     # 根据难度设定路径点数量
#     if difficulty == "Simple":
#         num_points = random.randint(2, 3)
#     elif difficulty == "Medium":
#         num_points = random.randint(4, 6)
#     else:  # Complex
#         num_points = random.randint(7, 10)
#     # 随机生成路径点坐标 (x, y) 和统一的飞行高度 z
#     waypoints = []
#     for _ in range(num_points):
#         x = random.randint(-50, 50)
#         y = random.randint(-50, 50)
#         z = random.randint(-20, -5)  # 高度在5~20米之间（负值向上）
#         waypoints.append((x, y, z))
#
#     # 如果任务复杂，进行简单路径优化（最近邻排序路径）
#     if difficulty == "Complex":
#         current = (0, 0)
#         optimized_path = []
#         remaining = waypoints.copy()
#         while remaining:
#             # 按与当前点距离排序，选择最近的下一个点
#             remaining.sort(key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
#             next_pt = remaining.pop(0)
#             optimized_path.append(next_pt)
#             current = (next_pt[0], next_pt[1])
#         waypoints = optimized_path
#
#     # 生成按顺序访问路径点的代码
#     for (x, y, z) in waypoints:
#         code_lines.append(f"client.moveToPositionAsync({x}, {y}, {z}, 5).join()")
#
#     # 所有路径点访问完毕后降落
#     code_lines.extend([
#         "client.hoverAsync().join()",
#         "time.sleep(2)",
#         "client.landAsync().join()",
#         "client.armDisarm(False)",
#         "client.enableApiControl(False)"
#     ])
#
#     # 生成任务描述文本
#     if difficulty == "Complex":
#         description = f"无人机复杂路径规划任务：计算优化路径访问{num_points}个目标点，依次飞行后降落。"
#     elif difficulty == "Medium":
#         description = f"无人机路径规划任务：按照预定顺序访问{num_points}个路径点，中途可悬停观察，最终降落。"
#     else:
#         description = f"无人机简单路径规划任务：依次飞往{num_points}个目标点，然后安全降落。"
#     return {"task_type": "路径规划", "difficulty": difficulty, "description": description, "code": "\n".join(code_lines)}
#
# # 生成“避障”任务
# def generate_obstacle_avoidance_task(difficulty):
#     code_lines = []
#     code_lines.extend([
#         "import airsim",
#         "client = airsim.MultirotorClient()",
#         "client.confirmConnection()",
#         "client.enableApiControl(True)",
#         "client.armDisarm(True)",
#         "client.takeoffAsync().join()"
#     ])
#     # 设置起点、终点和障碍物位置（模拟值）
#     start_alt = -10  # 起飞高度10米 (AirSim中负值表示向上)
#     target_x = random.randint(30, 60)
#     target_y = random.randint(-10, 10)
#     target_z = start_alt
#     obstacle_x = random.randint(10, 20)
#     obstacle_y = random.randint(-5, 5)
#     code_lines.append(f"target = ({target_x}, {target_y}, {target_z})  # 目标点坐标")
#     code_lines.append(f"obstacle = ({obstacle_x}, {obstacle_y})       # 障碍物所在平面位置")
#     code_lines.append("avoid_alt = -15  # 绕障时的飞行高度（15米）")
#
#     if difficulty == "Simple":
#         # 简单避障：在中途检查一次障碍并绕过
#         mid_x = target_x / 2
#         mid_y = target_y / 2
#         code_lines.append(f"client.moveToPositionAsync({mid_x}, {mid_y}, {start_alt}, 5).join()")
#         code_lines.append("# 检测与障碍的距离")
#         code_lines.append("dist = ((obstacle[0] - {mx})**2 + (obstacle[1] - {my})**2)**0.5".format(mx=mid_x, my=mid_y))
#         code_lines.append("if dist < 5:")
#         code_lines.append("    # 障碍过近，调整航线绕过障碍")
#         code_lines.append("    client.moveToPositionAsync(obstacle[0], obstacle[1] + 10, avoid_alt, 5).join()")
#         code_lines.append("client.moveToPositionAsync(target[0], target[1], target[2], 5).join()")
#         desc_text = "飞行途中检测到障碍则临时改变高度绕过"
#     elif difficulty == "Medium":
#         # 中等避障：分段飞行，多次检查障碍
#         code_lines.append("for i in range(5):")
#         code_lines.append("    # 将路径分为5段逐步飞行")
#         code_lines.append("    frac = (i + 1) / 5.0")
#         code_lines.append("    cur_x = target[0] * frac")
#         code_lines.append("    cur_y = target[1] * frac")
#         code_lines.append("    client.moveToPositionAsync(cur_x, cur_y, target[2], 4).join()")
#         code_lines.append("    dist = ((obstacle[0] - cur_x)**2 + (obstacle[1] - cur_y)**2)**0.5")
#         code_lines.append("    if dist < 5:")
#         code_lines.append("        # 障碍物接近，执行绕行：先上升再绕过")
#         code_lines.append("        client.moveToPositionAsync(cur_x, cur_y + 15, avoid_alt, 4).join()")
#         code_lines.append("        client.moveToPositionAsync(cur_x + 15, cur_y + 15, target[2], 4).join()")
#         code_lines.append("        break")
#         code_lines.append("client.moveToPositionAsync(target[0], target[1], target[2], 5).join()")
#         desc_text = "逐步前进并多次检测障碍，遇障碍时绕行"
#     else:
#         # 复杂避障：模拟持续扫描并路径重规划
#         code_lines.append("import math")
#         code_lines.append("curr_x, curr_y = 0, 0")
#         code_lines.append("path = []  # 将规划的路径点存储在列表")
#         code_lines.append("while math.hypot(target[0]-curr_x, target[1]-curr_y) > 5:")
#         code_lines.append("    # 朝目标前进小步长")
#         code_lines.append("    step_x = curr_x + 5 if target[0] > curr_x else curr_x - 5")
#         code_lines.append("    step_y = curr_y + 5 if target[1] > curr_y else curr_y - 5")
#         code_lines.append("    # 检测障碍物距离")
#         code_lines.append("    dist = math.hypot(obstacle[0] - step_x, obstacle[1] - step_y)")
#         code_lines.append("    if dist < 5:")
#         code_lines.append("        # 障碍物过近，修改路径：改变方向绕行")
#         code_lines.append("        step_y += 20  # 把路径偏移20单位绕过障碍")
#         code_lines.append("    path.append((step_x, step_y, target[2] if dist >= 5 else avoid_alt))")
#         code_lines.append("    curr_x, curr_y = step_x, step_y")
#         code_lines.append("# 按规划路径飞行")
#         code_lines.append("for px, py, pz in path:")
#         code_lines.append("    client.moveToPositionAsync(px, py, pz, 5).join()")
#         code_lines.append("client.moveToPositionAsync(target[0], target[1], target[2], 5).join()")
#         desc_text = "实时扫描前方障碍并动态调整路径绕行"
#
#     # 降落并释放控制
#     code_lines.extend([
#         "client.hoverAsync().join()",
#         "time.sleep(2)",
#         "client.landAsync().join()",
#         "client.armDisarm(False)",
#         "client.enableApiControl(False)"
#     ])
#     description = f"无人机避障任务：{desc_text}，最后安全降落。"
#     return {"task_type": "避障", "difficulty": difficulty, "description": description, "code": "\n".join(code_lines)}
#
# # 生成“目标搜索”任务
# def generate_target_search_task(difficulty):
#     code_lines = []
#     code_lines.extend([
#         "import airsim",
#         "client = airsim.MultirotorClient()",
#         "client.confirmConnection()",
#         "client.enableApiControl(True)",
#         "client.armDisarm(True)",
#         "client.takeoffAsync().join()"
#     ])
#     # 根据难度设定搜索区域大小和步进
#     if difficulty == "Simple":
#         area_size = 20   # 区域边长20米
#         step = 10        # 10米步进（扫描间隔）
#     elif difficulty == "Medium":
#         area_size = 50
#         step = 10
#     else:
#         area_size = 100
#         step = 20
#
#     code_lines.append("# 执行Z字形搜索覆盖指定区域")
#     code_lines.append("target_found = False")
#     code_lines.append(f"for j in range(0, {area_size}, {step}):")
#     code_lines.append(f"    y = j")
#     code_lines.append("    if j // {0} % 2 == 0:".format(step))  # 模拟奇偶行的方向切换
#     code_lines.append(f"        x_range = range(0, {area_size}, {step})")
#     code_lines.append("    else:")
#     code_lines.append(f"        x_range = range({area_size}, -1, -{step})")
#     code_lines.append("    for x in x_range:")
#     code_lines.append("        client.moveToPositionAsync(x, y, -10, 5).join()")
#     code_lines.append("        # 检查是否发现目标（这里可调用图像识别模型）")
#     code_lines.append("        if target_found:")
#     code_lines.append("            break")
#     code_lines.append("    if target_found:")
#     code_lines.append("        break")
#     code_lines.append("if target_found:")
#     code_lines.append("    client.hoverAsync().join()  # 发现目标后悬停定位")
#     code_lines.append("else:")
#     code_lines.append("    client.landAsync().join()   # 未找到目标则执行降落")
#     code_lines.extend([
#         "client.armDisarm(False)",
#         "client.enableApiControl(False)"
#     ])
#
#     # 生成描述文本
#     if difficulty == "Complex":
#         description = "无人机目标搜索任务：在广阔区域执行Z字形航线搜索，覆盖整个区域后定位目标。"
#     elif difficulty == "Medium":
#         description = "无人机目标搜索任务：在指定区域内逐行扫描目标，一旦发现立即悬停定位。"
#     else:
#         description = "无人机目标搜索任务：在小范围区域执行Z字形搜索，如发现目标则悬停等待。"
#     return {"task_type": "目标搜索", "difficulty": difficulty, "description": description, "code": "\n".join(code_lines)}
#
# # 按类别批量生成任务
# tasks = []
# for category in tqdm(categories):
#     if category == "Composite Instruction":
#         generator = generate_composite_task
#     elif category == "Path Planning":
#         generator = generate_path_planning_task
#     elif category == "Obstacle Avoidance":
#         generator = generate_obstacle_avoidance_task
#     else:  # Target Search
#         generator = generate_target_search_task
#
#     # 按照预定比例分配各难度数量
#     simple_count = int(tasks_per_category * difficulty_distribution["Simple"])
#     medium_count = int(tasks_per_category * difficulty_distribution["Medium"])
#     complex_count = tasks_per_category - simple_count - medium_count
#     difficulty_list = ["Simple"] * simple_count + ["Medium"] * medium_count + ["Complex"] * complex_count
#     random.shuffle(difficulty_list)  # 打乱顺序增加多样性
#
#     # 生成对应数量的任务
#     for diff in difficulty_list:
#         task = generator(diff)
#         tasks.append(task)
#
# # 保存任务数据集为 JSONL 格式文件
# with open("airsim_tasks.jsonl", "w", encoding="utf-8") as f:
#     for task in tasks:
#         # 使用ensure_ascii=False以确保中文正常写入
#         f.write(json.dumps(task, ensure_ascii=False) + "\n")
#
import json

# 需要分割的原始JSONL文件路径
input_file = "airsim_tasks.jsonl"

# 存放每种任务的文件路径
output_files = {
    "Composite Instruction": "composite_instruction_with_hint.jsonl",
    "Path Planning": "path_planning.jsonl",
    "Obstacle Avoidance": "obstacle_avoidance.jsonl",
    "Target Search": "target_search.jsonl"
}

# 打开所有输出文件
file_handles = {task_type: open(filename, 'w', encoding='utf-8')
                for task_type, filename in output_files.items()}

# 逐行读取并根据task_type写入对应文件
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        task_type = data.get("task_type")
        if task_type in file_handles:
            file_handles[task_type].write(json.dumps(data, ensure_ascii=False) + "\n")

# 关闭所有文件
for handle in file_handles.values():
    handle.close()

print("任务分割完成。")
