import airsim
import cv2
import numpy as np
import PIL.Image
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import warnings
import re

warnings.filterwarnings("ignore")

# Gemini API 配置
GOOGLE_API_KEY = "AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88"
genai.configure(api_key=GOOGLE_API_KEY)

# 加载FAISS知识库（RAG）
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vector_store = FAISS.load_local("../../scripts/api_learning/airsim_api_faiss",
                                embeddings, allow_dangerous_deserialization=True)
llm_rag = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp", google_api_key=GOOGLE_API_KEY)
rag_chain = RetrievalQA.from_chain_type(llm=llm_rag, retriever=vector_store.as_retriever(), chain_type="stuff")

# VLM模型
vlm_model = genai.GenerativeModel('gemini-2.0-pro-exp')

# AirSim 客户端
client = airsim.MultirotorClient()
client.confirmConnection()

camera_name = "0"
image_type = airsim.ImageType.Scene

# 辅助函数
def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks).strip()
        if full_code.startswith("python"):
            full_code = full_code[7:].strip()
        return full_code
    return None

# Gemini+RAG主交互
def query_rag_gemini(question: str) -> str:
    response = rag_chain.run(question)
    return response

# Gemini-VLM视觉模型调用（按需调用）
def query_vlm(command: str, image: PIL.Image) -> str:
    response = vlm_model.generate_content([command, image])
    response.resolve()
    return response.text

# 主交互循环
while True:
    command = input("\nEnter command (!quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break

    # 首先尝试通过Gemini+RAG处理
    response = query_rag_gemini(command)
    print("\nGemini+RAG Response:", response)

    # 检查Gemini+RAG回复中是否需要视觉辅助（例如包含特定关键词）
    if "视觉" in response or "无法确定" in response or "需要图像" in response:
        print("\n正在调用视觉模型确认...")

        # 获取摄像头实时画面
        raw_image = client.simGetImage(camera_name, image_type)
        if raw_image:
            np_img = np.frombuffer(raw_image, dtype=np.uint8)
            img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(img_rgb)

            # 调用VLM确认视觉信息
            vlm_response = query_vlm(command, img_pil)
            print("\nGemini-VLM Response:", vlm_response)

            # 尝试从VLM的回答中提取代码
            code = extract_python_code(vlm_response)
            if code:
                print("\nExtracted Code from VLM:\n", code)
                try:
                    exec(code)
                except Exception as e:
                    print("\nCode Execution Error:", e)
            else:
                print("\nNo executable code found from VLM.")
        else:
            print("\nWarning: 无法获取图像数据。")

    else:
        # 尝试从Gemini+RAG的回答中提取代码
        code = extract_python_code(response)
        if code:
            print("\nExtracted Code from Gemini+RAG:\n", code)
            try:
                exec(code)
            except Exception as e:
                print("\nCode Execution Error:", e)
        else:
            print("\nNo executable code found from Gemini+RAG.")