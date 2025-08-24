import os
import re
import warnings
from typing import TypedDict, Optional, List, Any
from PIL import Image

warnings.filterwarnings('ignore')

from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import Tool, AgentExecutor
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import render_text_description_and_args
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool



import google.generativeai as genai

from langgraph.graph import StateGraph, END

from detection_module import *


# --------------- API Key 初始化 ---------------
os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88' # 谷歌主号
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyCn3xz6FWTR7KCrcTRPojjxvMTHheBUkeQ' # 谷歌：zihan1022@gmail.com
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyDLcbDfYYofvmdL1cGmpA7wZoy-sCH4SVc' # 谷歌：zzh 东大
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyAx9-Dpt4w2CErLYjIFvrqO_h8rlN60aSs' # 谷歌：sjz
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyAJXUOYxiP85Z4c5wY3r7i9zpK7s_-3SpY' # 谷歌：YAV
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyANKgku-kmhcYiFBMyr_D4PCzWbEukNEA0' # 谷歌：IAN blue
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBVykq8bzBJiKdAuBi9Oj84a2MxxkOYWiw' # 谷歌：zihan1576518
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用 LangSmith Tracing
os.environ["LANGCHAIN_PROJECT"] = "DroneControlGraph"  # 指定项目名称，可自定义

class GraphState(TypedDict):
    user_input: str                     # 用户输入的文本
    image: Optional[Image.Image]        # 用户输入的图像（可选）
    tracker_results: Optional[Any]     # tracker_llm 的检测结果（深度、偏向角等）
    coder_output: Optional[Any]          # coder_agent_executor 生成的控制代码
    execution_result: Optional[Any]     # 执行代码后的结果
    response: str                       # 最终返回给用户的响应

global_namespace = globals()

coder_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)
coder_vectorstore = FAISS.load_local(
    "airsim_1.8.1_windows.db",
    coder_embedding,
    allow_dangerous_deserialization=True
)
coder_retriever = coder_vectorstore.as_retriever()
coder_websearch_tool = TavilySearchResults(k=5)
coder_llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp")

# interactor_llm = genai.GenerativeModel('gemini-2.0-flash')

@tool
def vector_search_tool(query: str) -> str:
    """Tool for searching the vector store."""
    qa_chain = RetrievalQA.from_chain_type(llm=coder_llm, retriever=coder_retriever)
    return qa_chain.invoke(query)

@tool
def web_search_tool(query: str) -> str:
    """Tool for performing web search."""
    return coder_websearch_tool.invoke(query)

coder_tools = [
    Tool(
        name="VectorStoreSearch",
        func=vector_search_tool,
        description="Use this to search the vector store for information."
    ),
    Tool(
        name="WebSearch",
        func=web_search_tool,
        description="Use this to perform a web search for information."
    ),
]

def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)

    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None

def code_executor(code):
    # 尝试从输出中提取 Python 代码块
    # code = extract_python_code(response['output'])

    if code is not None:
        print("Extracted code:\n")
        print(code)
        try:
            exec(code, global_namespace)
            if 'main' in global_namespace:
                global_namespace['main']()
        except Exception as e:
            print("Error executing code:", e)

coder_system_prompt = """
You are an intelligent assistant that helps me control the drone in AirSim and complete a series of tasks.
When I ask you to do something, you are supposed to give me Python code that is needed to achieve that task using AirSim and then an explanation of what that code does.
You are only allowed to use the airsim functions in the version of 1.8.1 under the Windows OS, so you are not to use any other hypothetical functions that you think might exist.
You can use simple Python functions from libraries such as math, numpy and cv2.
Before embarking on any mission, you should determine the state of the drone firstly, so that it can be used for subsequent missions.
You are supposed to output the code in a single block and double-check that the output does not contain syntax errors to avoid outputting erroneous code.
If the instruction contains a situation that needs to be located, you should check context first, instead of trying to use the AirSim API for detecting scene objects.

Follow these rules:
1. Always try the \"VectorStoreSearch\" tool first.
2. Only use \"WebSearch\" if the vector store does not contain the required information.
3. Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
4. Valid "action" values: "Final Answer" or {tool_names}
5. Provide only ONE action per JSON_BLOB
"""
coder_human_prompt = """{input}  
{agent_scratchpad}  
(reminder to always respond in a JSON blob)"""
tracker_prompt = "Please analyze the drone scene and return detected objects with their distances and yaw angles."

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", coder_system_prompt),
    ("human", coder_human_prompt),
])

coder_prompt = coder_prompt.partial(
    tools=render_text_description_and_args(list(coder_tools)),
    tool_names=", ".join([t.name for t in coder_tools]),
)

coder_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | coder_prompt
    | coder_llm
    | JSONAgentOutputParser()
)

coder_agent_executor = AgentExecutor(
    agent=coder_chain,
    tools=coder_tools,
    handle_parsing_errors=True,
    verbose=True
)


def tracker_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    image = state.get("image", get_airsim_scene())

    content = [user_input]
    if image:
        content.append(image)

    s = time.time()
    response = tracker_llm.generate_content(content)
    end = time.time()
    print(f"交互用时:{end-s}")

    # 简单询问直接返回
    if not any(keyword in user_input.lower() for keyword in ["检测", "距离", "偏向角", "控制", "移动"]):
        state["response"] = response.text
        return state

    # 检查是否需要目标检测
    if any(keyword in user_input.lower() for keyword in ["检测", "距离", "偏向角"]):
        print("正在调用 process_scene_with_detection 处理请求...")

        response = tracker_llm.generate_content([tracker_prompt], tools={"function_declarations": tracker_tool})
        # 检查Gemini响应，调用实际函数
        if response.candidates[0].content.parts[0].function_call.name == "estimate_depth_yaw":
            scene_analysis_results = estimate_depth_yaw()

            state['tracker_results'] = scene_analysis_results
            # print("tracker: ", state)
            return state

    # 需要控制无人机
    if any(keyword in user_input.lower() for keyword in ["控制", "移动"]):
        state["response"] = "正在委托 coder_agent_executor 生成控制代码..."
        return state

    state["response"] = "无法识别的请求"
    return state


def coder_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    tracker_results = state.get("tracker_results")

    # print("coder: ", state)

    # 构建上下文，包含 tracker 的结果（如果有）
    context = f"指令: {user_input}\n"
    if tracker_results and "error" not in tracker_results:
        context += f"目标检测结果: {tracker_results}\n"

    # 调用 coder_agent_executor
    coder_response = coder_agent_executor.invoke({"input": context})

    # 提取并执行代码
    code_time = time.time()
    code_executor(extract_python_code(coder_response["output"]))
    code_end_time = time.time()
    print(f'代码执行用时:{code_end_time - code_time}')
    state["coder_output"] = coder_response['output']
    # state["execution_result"] = execution_result
    state["response"] = f"控制代码已生成并执行"

    return state


# 条件路由函数
def route_task(state: GraphState) -> str:
    user_input = state["user_input"].lower()
    if any(keyword in user_input for keyword in ["控制", "移动"]):
        return "coder_node"
    return END

workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("tracker_node", tracker_node)
workflow.add_node("coder_node", coder_node)

# 设置入口
workflow.set_entry_point("tracker_node")

# 添加条件边
workflow.add_conditional_edges(
    "tracker_node",
    route_task,
    {
        "coder_node": "coder_node",
        END: END
    }
)

# coder_node 完成后结束
workflow.add_edge("coder_node", END)

# 编译图
app = workflow.compile()

state = {
    "user_input": None,
    "image": None,
    "tracker_results": None,
    "code_output": None,
    "execution_result": None,
    "response": ""
}

# --------------- AirSim 初始化 ---------------
client = airsim.MultirotorClient()
client.confirmConnection()
camera_name = "0"
image_type = airsim.ImageType.Scene

while True:
    command = input("Enter your command (!exit or !quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break

    state["user_input"] = command
    state["image"] = get_airsim_scene()

    start = time.time()
    state = app.invoke(state)

    print(state['response'])
    end = time.time()
    print(f'总用时:{end - start}')

    # response = coder_agent_executor.invoke({"input": command})
    #
    # print("Gemini output:", response['output'])
    #
    # code = extract_python_code(response['output'])
    # code_executor(code)