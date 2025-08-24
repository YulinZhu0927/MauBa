import io
from typing import TypedDict, Annotated
import operator
from PIL import Image
from langgraph.graph import StateGraph, END
from google.generativeai import GenerativeModel, configure

# 配置Gemini API密钥
configure(api_key="AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88")  # 替换为你的API密钥
gemini_model = GenerativeModel("gemini-2.0-pro-exp")  # 支持图像的Gemini模型


# 定义状态
class ImageState(TypedDict):
    image: Image.Image  # 存储PIL Image对象
    messages: list[str]  # 存储Gemini的响应
    current_agent: str  # 当前智能体


# 定义调用Gemini的节点
def gemini_agent(state: ImageState) -> ImageState:
    # print("Gemini Agent 执行")
    # print("进入时状态:", state["messages"])
    pil_image = state["image"]
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    image_data = img_byte_arr.getvalue()

    prompt = "请描述这张图片的内容。"
    response = gemini_model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": image_data}]
    )

    # print("Gemini原始响应:", repr(response.text))
    message = f"Gemini: {response.text}"
    if message not in state["messages"]:
        state["messages"].append(message)
    # print("退出时状态:", state["messages"])
    state["current_agent"] = "end"
    return state


# 构建图
graph = StateGraph(ImageState)
graph.add_node("gemini", gemini_agent)
graph.add_edge("gemini", END)
graph.set_entry_point("gemini")

# 编译图
runnable = graph.compile()

# 创建示例PIL对象（假设从文件加载）
image_path = "test.jpg"  # 替换为实际路径
pil_image = Image.open(image_path).convert("RGB")  # 转换为RGB格式

# 输入示例
initial_state = {
    "image": pil_image,
    "messages": [],
    "current_agent": "gemini"
}

# 运行并获取结果
result = runnable.invoke(initial_state)
print("Gemini输出:", result["messages"])