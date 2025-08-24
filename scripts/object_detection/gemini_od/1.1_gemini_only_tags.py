import google.generativeai as genai
import re
from PIL import Image
import cv2
import numpy as np

genai.configure(api_key='AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88')
gemini = genai.GenerativeModel('gemini-2.0-pro-exp')

input_image = r"img_Drone__0_1742387833154831000.png"
img = Image.open(input_image)

response = gemini.generate_content([
    img,
    (
        "Return tags for all objects as the following format in the image you can see."
        "e.g., you see a cat and 2 dogs, then response: a cat. two dogs."
        "Note that all the tags should be lowercased and end with a dot."
    ),
])

result = response.text
print(result)