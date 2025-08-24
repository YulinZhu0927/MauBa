import os

import PIL
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64, httpx

os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

# Create an instance of the LLM, using the 'gemini-pro' model with a specified creativity level
llm = ChatGoogleGenerativeAI(model='gemini-2.0-pro-exp')
image_data = base64.b64encode(PIL.Image.open(r'./img/test.jpg')).decode("utf-8")

# Send a creative prompt to the LLM
response = llm.invoke([
    HumanMessage(
        content=[{
            "type": "text",
            "text": "What's in this picture?",
        }, {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        }]
    )
])
print(response.content)