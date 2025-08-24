import json
import os
import re
import warnings
from typing import TypedDict, Optional, List, Any, Annotated
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
from langgraph.graph.message import add_messages

from detection_module import *

# --------------- API Key 初始化 ---------------
os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCn3xz6FWTR7KCrcTRPojjxvMTHheBUkeQ'
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DroneControlGraph"

# 定义消息状态
class MessagesState(TypedDict):
    messages: Annotated[List[Any], add_messages]  # 使用 add_messages 管理消息列表
    image: Optional[Image.Image]                  # 用户输入的图像（可选）
    next: Optional[str]                           # 下一个节点

# 全局命名空间
global_namespace = globals()

# 初始化嵌入和向量存储
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
supervisor_llm = genai.GenerativeModel('gemini-2.0-pro-exp')

# 定义工具
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

# 提取代码的辅助函数
def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("python"):
            full_code = full_code[7:]
        return full_code
    return None

def extract_json_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("json"):
            full_code = full_code[5:]
        return full_code
    return None

def code_executor(code):
    if code:
        print("Extracted code:\n", code)
        try:
            exec(code, global_namespace)
            if 'main' in global_namespace:
                global_namespace['main']()
        except Exception as e:
            print("Error executing code:", e)

# Prompt 定义
coder_system_prompt = """
You are an intelligent assistant that helps me control the drone in AirSim and complete a series of tasks.
When I ask you to do something, you are supposed to give me Python code that is needed to achieve that task using AirSim and then an explanation of what that code does.
You are only allowed to use the airsim functions in the version of 1.8.1 under the Windows OS.
You can use simple Python functions from libraries such as math, numpy and cv2.
Before embarking on any mission, you should determine the state of the drone firstly.
You are supposed to output the code in a single block and double-check that the output does not contain syntax errors.
If the instruction contains a situation that needs to be located, you should check context first.
Follow these rules:
1. Always try the "VectorStoreSearch" tool first.
2. Only use "WebSearch" if the vector store does not contain the required information.
3. Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
4. Valid "action" values: "Final Answer" or {tool_names}
5. Provide only ONE action per JSON_BLOB
"""
coder_human_prompt = """{input}  
{agent_scratchpad}  
(reminder to always respond in a JSON blob)"""
tracker_prompt = "Please analyze the drone scene and return detected objects with their distances and yaw angles."
supervisor_prompt = """
You are an intelligent assistant that helps me control the drone in AirSim and complete a series of tasks.
You are a supervisor agent coordinating two other agents: 'tracker' and 'coder'.
The role of 'tracker' is: 
    - Object detection, and
    - return the current distance and bias Angle between all the targets seen in the scene and the drone.
The role of 'coder' is: 
    - Generate the overall control code according to the instructions.
Your role is to:
    1. Receive user's instruction and analyze it to determine to which agent work should be delegated.
    2. Interact with the 'tracker' agent to get scene analysis results.
    3. Aggregate useful information and pass it to the 'coder' agent.
    4. Ensure the context is clear, structured and consistent between agents.
    5. Run the code, and return the final response to the user.
To delegate work, return a json string with two keys: delegation and output, as this schema: ```json{'delegation': str, 'output': str}```.
If it's a simple interaction, delegation is 'supervisor_node', and 'output' is your response.
If delegating, use 'tracker_node' or 'coder_node' for delegation, and 'output' is the command or aggregated info.
NOTICE: 
    1. If the detected object differs from the user's instruction, confirm with the user.
    2. If the object isn't in view, delegate 'coder' to rotate the view.
"""
chat_history = [{"role": "user", "parts": [supervisor_prompt]}]

# 配置 Coder Agent
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

# 节点定义
def supervisor_node(state: MessagesState) -> MessagesState:
    last_message = state["messages"][-1].content
    image = state.get("image", get_airsim_scene())
    content = [last_message]
    if image:
        content.append(image)

    response = supervisor_llm.generate_content(chat_history + [{"role": "user", "parts": content}])
    chat_history.append({"role": "user", "parts": content})
    chat_history.append({"role": "model", "parts": [response.text]})

    response_json = json.loads(str(extract_json_code(response.text)))
    which_executor = response_json['delegation']
    output = response_json['output']

    if which_executor == 'supervisor_node':
        return {"messages": [{"role": "assistant", "content": output}], "next": "end"}
    elif which_executor == 'tracker_node':
        return {"messages": [{"role": "user", "content": output}], "next": "tracker_node"}
    elif which_executor == 'coder_node':
        return {"messages": [{"role": "user", "content": output}], "next": "coder_node"}
    return {"messages": [{"role": "assistant", "content": output}], "next": "end"}

def tracker_node(state: MessagesState) -> MessagesState:
    print("已将工作委托给tracker...")
    user_input = state["messages"][-1].content
    image = state.get("image", get_airsim_scene())
    content = [tracker_prompt, user_input]
    if image:
        content.append(image)

    response = tracker_llm.generate_content(content, tools={"function_declarations": tracker_tool})
    if response.candidates[0].content.parts[0].function_call.name == "estimate_depth_yaw":
        scene_analysis_results = estimate_depth_yaw()
        return {
            "messages": [{"role": "assistant", "content": json.dumps(scene_analysis_results)}],
            "next": "supervisor_node"
        }
    return {
        "messages": [{"role": "assistant", "content": "无法识别的请求"}],
        "next": "supervisor_node"
    }

def coder_node(state: MessagesState) -> MessagesState:
    print("已将工作委托给coder...")
    user_input = state["messages"][-1].content
    coder_response = coder_agent_executor.invoke({"input": user_input})
    code = extract_python_code(coder_response["output"])
    code_executor(code)
    return {
        "messages": [{"role": "assistant", "content": "控制代码已生成并执行"}],
        "next": "supervisor_node"
    }

# 路由函数
def route_task(state: MessagesState) -> str:
    next_node = state.get("next", "end")
    return next_node if next_node in ["supervisor_node", "tracker_node", "coder_node", "end"] else "end"

# 创建工作流
workflow = StateGraph(MessagesState)
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("tracker_node", tracker_node)
workflow.add_node("coder_node", coder_node)
workflow.set_entry_point("supervisor_node")
workflow.add_conditional_edges(
    "supervisor_node",
    route_task,
    {
        "supervisor_node": "supervisor_node",
        "tracker_node": "tracker_node",
        "coder_node": "coder_node",
        "end": END
    }
)
workflow.add_edge("tracker_node", "supervisor_node")
workflow.add_edge("coder_node", "supervisor_node")
app = workflow.compile()

# 主循环
client = airsim.MultirotorClient()
client.confirmConnection()
camera_name = "0"
image_type = airsim.ImageType.Scene

while True:
    command = input("Enter your command (!exit or !quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break

    initial_state = {
        "messages": [{"role": "user", "content": command}],
        "image": get_airsim_scene(),
        "next": "supervisor_node"
    }
    state = app.invoke(initial_state)
    print(state["messages"][-1].content)