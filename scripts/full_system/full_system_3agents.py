import json
import os
import re
import warnings
from typing import TypedDict, Optional, List, Any
from PIL import Image
import time

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
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyAZnKIzRkILIF_A3ZVxeFX8HtiFxPhAB-Q' # 谷歌：zyl 菠萝
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyDLcbDfYYofvmdL1cGmpA7wZoy-sCH4SVc' # 谷歌：zzh 东大
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBVykq8bzBJiKdAuBi9Oj84a2MxxkOYWiw' # 谷歌：zihan1576518
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用 LangSmith Tracing
os.environ["LANGCHAIN_PROJECT"] = "DroneControlGraph"  # 指定项目名称，可自定义

class GraphState(TypedDict):
    user_input: str                     # 用户输入的文本
    image: Optional[Image.Image]        # 用户输入的图像（可选）
    tracker_output: Optional[Any]      # tracker_llm 的检测结果（深度、偏向角等）
    coder_output: Optional[Any]         # coder_agent_executor 生成的控制代码
    supervisor_output: Optional[Any]    # supervisor的决策
    execution_result: Optional[Any]     # 执行代码后的结果
    response: str                       # 最终返回给用户的响应
    next: str                           # 下一个节点是谁

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

supervisor_llm = genai.GenerativeModel('gemini-2.0-flash')

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
    # Tool(
    #     name="VectorStoreSearch",
    #     func=vector_search_tool,
    #     description="Use this to search the vector store for information."
    # ),
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

def extract_json_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)

    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("json"):
            full_code = full_code[5:]

        return full_code
    else:
        return None

@tool
def code_executor(content):
    """
        The AirSim drone control Python code included in the Coder Agent's answer is extracted and executed.

    Args:
        content (str): Contains the text of the code that the function will attempt to extract Python code from and execute.
    """
    # 尝试从输出中提取 Python 代码块
    code = extract_python_code(content)

    if code is not None:
        try:
            exec(code, global_namespace)
            # if 'main' in global_namespace:
            #     global_namespace['main']()
        except Exception as e:
            print("Error executing code:", e)

supervisor_tools = [
    Tool(
        name="CodeExecutor",
        func=code_executor,
        description="Use this to execute code obtained from the Coder Agent."
    ),
]

def get_prompts(path):
    with open(path) as f:
        prompt = f.read()
        return prompt

coder_system_prompt = get_prompts(r'./prompts/coder_system_prompt.txt')
coder_human_prompt = get_prompts(r'./prompts/coder_human_prompt.txt')
tracker_prompt = get_prompts(r'./prompts/tracker_prompt.txt')
supervisor_prompt = get_prompts(r'./prompts/supervisor_prompt.txt')
coder_exec_prompt = get_prompts(r'./prompts/coder_exec_prompt.txt')

code_executor_tool_config = [{
    "name": "code_executor",
    "description": "To execute the code generated by Coder Agent, pass 'coder_output' as an argument.",
    "parameters": {
         "type": "object",
         "properties": {
             "coder_output": {
                 "type": "string",
                 "description": "Contains the code and description information produced by the Coder Agent."
             }
         },
         "required": ["coder_output"]
    }
}]

chat_history = [{"role": "user", "parts": [supervisor_prompt]}]

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", coder_system_prompt),
    ("human", coder_human_prompt),
]).partial(
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


def supervisor_node(state: GraphState) -> GraphState:
    tracker_output = ""
    coder_output = ""
    if state['tracker_output'] is not None:
        tracker_output = str(state['tracker_output'])
        state['tracker_output'] = None

    if state['coder_output'] is not None:
        coder_output = str(state['coder_output'])
        print(state) # 拿到代码了
        state['coder_output'] = None

    state['next'] = 'end'

    user_input = state["user_input"]
    image = state.get("image", get_airsim_scene())
    content = [user_input, tracker_output, coder_output]
    if image:
        content.append(image)

    if coder_output != "":
        code_start = time.time()
        response = supervisor_llm.generate_content(
            [coder_exec_prompt, coder_output], tools={"function_declarations": code_executor_tool_config}
        )

        if not response.candidates and not response.candidates[0].content.parts:
            print("No valid candidate or parts found")
            return state

        # print(response.candidates[0].content.parts[0].function_call.name)
        # 检查Gemini响应，调用实际函数
        if response.candidates[0].content.parts[0].function_call.name == "code_executor":
            code_executor(coder_output)
            state['next'] = 'supervisor_node'
            code_end = time.time()
            print(f"代码执行时间:{code_end-code_start}")
            return state

    start_time = time.time()
    response = supervisor_llm.generate_content(chat_history + [{"role": "user", "parts": content}])

    # 更新聊天历史
    chat_history.append({"role": "user", "parts": content})
    chat_history.append({"role": "model", "parts": [response.text]})

    response_json = json.loads(
        str(extract_json_code(str(response.text)))
    )
    print(response_json['output'])
    end_time = time.time()
    print(f"交互时间:{end_time-start_time}")

    which_executor = response_json['delegation']

    state['supervisor_output'] = response_json['output']
    if which_executor == 'tracker_node':
        state['next'] = 'tracker_node'
        return state
    elif which_executor == 'coder_node':
        state['user_input'] += "\nOutput from Tracker Agent:" + tracker_output
        state['next'] = 'coder_node'
        return state
    else:
        state['next'] = 'end'
        return state


def tracker_node(state: GraphState) -> GraphState:
    print("已将工作委托给tracker...")

    user_input = state["user_input"]
    image = state.get("image", get_airsim_scene())

    content = [user_input]
    if image:
        content.append(image)

    print("正在调用 process_scene_with_detection 处理请求...")

    response = tracker_llm.generate_content([tracker_prompt], tools={"function_declarations": tracker_tool})
    # 检查Gemini响应，调用实际函数
    if response.candidates[0].content.parts[0].function_call.name == "estimate_depth_yaw":
        scene_analysis_results = estimate_depth_yaw()

        state['tracker_output'] = scene_analysis_results
        state['next'] = 'supervisor_node'
        # print("tracker: ", state)
        return state

    state["tracker_output"] = "无法识别的请求"
    state['next'] = 'supervisor_node'
    return state


def coder_node(state: GraphState) -> GraphState:
    print("已将工作委托给coder...")
    user_input = state["user_input"]

    # 调用 coder_agent_executor
    coder_response = coder_agent_executor.invoke({"input": user_input})
    state["coder_output"] = coder_response['output']
    state['next'] = 'supervisor_node'

    return state


# 条件路由函数
def route_task(state: GraphState) -> str:
    next_node = state.get("next", "end")
    return next_node if next_node in ["supervisor_node", "tracker_node", "coder_node", "end"] else "end"

workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("tracker_node", tracker_node)
workflow.add_node("coder_node", coder_node)

# 设置入口
workflow.set_entry_point("supervisor_node")

# 添加条件边
workflow.add_conditional_edges(
    "supervisor_node",
    route_task,
    {
        "supervisor_node": END,
        "tracker_node": "tracker_node",
        "coder_node": "coder_node",
        "end": END
    }
)

# tracker_node 和 coder_node 执行完后固定返回 supervisor_node
workflow.add_edge("tracker_node", "supervisor_node")
workflow.add_edge("coder_node", "supervisor_node")

# 编译图
app = workflow.compile()

# class GraphState(TypedDict):
#     user_input: str                     # 用户输入的文本
#     image: Optional[Image.Image]        # 用户输入的图像（可选）
#     tracker_output: Optional[Any]      # tracker_llm 的检测结果（深度、偏向角等）
#     coder_output: Optional[Any]         # coder_agent_executor 生成的控制代码
#     supervisor_output: Optional[Any]    # supervisor的决策
#     execution_result: Optional[Any]     # 执行代码后的结果
#     response: str                       # 最终返回给用户的响应
#     next: str                           # 下一个节点是谁

state = {
    "user_input": None,
    "image": None,
    "tracker_output": None,
    "coder_output": None,
    "supervisor_output": None,
    "execution_result": None,
    "response": "",
    "next": "supervisor_node"
}

# --------------- AirSim 初始化 ---------------
# client = airsim.MultirotorClient()
# client.confirmConnection()
camera_name = "0"
image_type = airsim.ImageType.Scene

while True:
    command = input("Enter your command (!exit or !quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break
    if command.lower() == 'r':
        client.reset()
        continue

    state["user_input"] = command
    state["image"] = get_airsim_scene()

    start = time.time()
    state = app.invoke(state)
    end = time.time()
    print(f'总用时:{end-start}')

    # print(state['response'])

    # response = coder_agent_executor.invoke({"input": command})
    #
    # print("Gemini output:", response['output'])
    #
    # code = extract_python_code(response['output'])
    # code_executor(code)