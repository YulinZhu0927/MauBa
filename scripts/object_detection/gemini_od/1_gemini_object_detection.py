import google.generativeai as genai
import re
from PIL import Image
import cv2
import numpy as np

genai.configure(api_key='AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88')

# import pprint
# for model in genai.list_models():
#     pprint.pprint(model)

gemini = genai.GenerativeModel('gemini-1.5-pro')

input_image = r"img_Drone__0_1742387833154831000.png"
img = Image.open(input_image)

response = gemini.generate_content([
    img,
    (
        "Return bounding boxes for all objects in the image in the following format as"
        " a list. \n [ymin, xmin, ymax, xmax, object_name]. If there are more than one object, return separate lists for each object"
    ),
])

result = response.text

def parse_bounding_box(response):
    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response)

    # Convert each group into a list of integers and labels.
    parsed_boxes = []
    for box in bounding_boxes:
        parts = box.split(',')
        numbers = list(map(int, parts[:-1]))
        label = parts[-1].strip()
        parsed_boxes.append((numbers, label))

    # Return the list of bounding boxes with their labels.
    return parsed_boxes

bounding_box = parse_bounding_box(result)

label_colors = {}

def draw_bounding_boxes(image, bounding_boxes_with_labels):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = np.array(image)

    for bounding_box, label in bounding_boxes_with_labels:

        # Normalize the bounding box coordinates.
        width, height = image.shape[1], image.shape[0]
        ymin, xmin, ymax, xmax = bounding_box
        x1 = int(xmin / 1000 * width)
        y1 = int(ymin / 1000 * height)
        x2 = int(xmax / 1000 * width)
        y2 = int(ymax / 1000 * height)

        if label not in label_colors:
            color = np.random.randint(0, 256, (3,)).tolist()
            label_colors[label] = color
        else:
            color = label_colors[label]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        box_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 5
        text_bg_x2 = x1 + text_size[0] + 8
        text_bg_y2 = y1


        cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

    image = Image.fromarray(image)
    return image

output = draw_bounding_boxes(img, bounding_box)
output.show()

print(result)