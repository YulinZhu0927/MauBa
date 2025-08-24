import numpy as np
import time

import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import warnings
warnings.filterwarnings('ignore')

# model_id = "IDEA-Research/grounding-dino-base"
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, disable_custom_kernels=True).to(device)

input_image = r"img_Drone__0_1742387833154831000.png"
# input_image = r"cat.jpg"
img = Image.open(input_image).convert("RGB")
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "blue car."

inputs = processor(images=img, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

statr_time = time.time()
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[img.size[::-1]]
)
end_time = time.time()

detected = results[0]

boxes = detected["boxes"]  # [N, 4]  (x1, y1, x2, y2)
scores = detected["scores"]  # [N]
labels = detected["labels"]  # [N]  (对零样本检测而言，通常是输入的文本query对应索引)

# 将PIL图像转为numpy数组 (H, W, 3)
draw_image = np.array(img.copy())

# BGR绘制
for box, score, label_idx in zip(boxes, scores, labels):
    box = box.tolist()
    x1, y1, x2, y2 = map(int, box)

    # 随机或固定一种颜色
    color = (0, 255, 0)

    # 画矩形框
    cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, thickness=2)

    # 生成标签文本；此示例仅有一个文本 "a car."，
    # 如果多文本，可以根据 label_idx 确定是哪条提示
    text_str = f"{label_idx}: {score:.2f}"

    # 在框上方写上标签和置信度
    cv2.putText(draw_image, text_str, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

# 转回PIL图像
annotated_img = Image.fromarray(draw_image)

# 显示可视化结果
annotated_img.show()
print("Total time: ", end_time - statr_time)