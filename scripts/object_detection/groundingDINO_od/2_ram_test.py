import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image = r"img_Drone__0_1742387833154831000.png"
image = r"cat.jpg"

im = Image.open(image)
# plt.imshow(im)
# plt.show()
model = ram(pretrained=r'../../../models/ram_swin_large_14m.pth',
                         image_size=384,
                         vit='swin_l')
model.eval()

model = model.to(device)

# 获取转换管道
transform = get_transform()

image = transform(Image.open(image)).unsqueeze(0).to(device)

res = inference(image, model)
print("Image Tags: ", res[0])
print("图像标签: ", res[1])

tags = [tag.strip().lower() for tag in res[0].split('|') if tag.strip()]
tags = [tag if tag.endswith('.') else tag + '.' for tag in tags]
text = ' '.join(tags)

print(text)
