import os
import re
import cv2
import numpy as np
import time
import math

import airsim
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import render_text_description_and_args
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import Tool, AgentExecutor
from langchain.tools import tool



os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                   encode_kwargs={"normalize_embeddings": True})

# api_vectorstore = FAISS.load_local("airsim_1.8.1_windows.db",
#                                           embeddings, allow_dangerous_deserialization=True)
vis_vectorstore = FAISS.load_local("airsim_vision.db",
                                          embeddings, allow_dangerous_deserialization=True)
# api_retriever = api_vectorstore.as_retriever()
vis_retriever = vis_vectorstore.as_retriever()

# 使用EnsembleRetriever，加权组合两个检索器
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[api_retriever, vis_retriever],
#     weights=[0.5, 0.5]  # 权重可调整，例如[0.7, 0.3]
# )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp")

web_search_tool = TavilySearchResults(k=10)

# create tool call for vector search and web search
@tool
def vector_search_tool(query: str) -> str:
    """Tool for searching the vector store."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vis_retriever)
    return qa_chain.invoke(query)


@tool
def web_search_tool_func(query: str) -> str:
    """Tool for performing web search."""
    return web_search_tool.invoke(query)

# define tools for the agent

tools = [
    Tool(
        name="VectorStoreSearch",
        func=vector_search_tool,
        description="Use this to search the vector store for information."
    ),
    Tool(
        name="WebSearch",
        func=web_search_tool_func,
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
            exec(code)
        except Exception as e:
            print("Error executing code:", e)

# define system prompt
system_prompt = """
You are an intelligent assistant that helps me control the drone in AirSim and complete a series of tasks.
Your primary task is to generate code that obtains the current distance and yaw angle of the object I wish to detect in the scene, assisting other assistants in accurately navigating to the predetermined position.

Important Constraints:
- For detecting objects and measuring their distance and yaw angle, you MUST use ONLY the predefined function: get_object_info_by_color.
- You are explicitly NOT allowed to use other AirSim APIs (such as simGetObjectPose, simGetSegmentationImage, etc.) or hypothetical functions you think might exist.
- You may use standard Python libraries such as math, numpy, and cv2.
- Ensure the generated code works with AirSim version 1.8.1 under Windows OS.

Guidelines for Querying VectorStoreSearch:
- Always explicitly include "get_object_info_by_color" in your search queries when retrieving information.
- Never include irrelevant keywords such as "segmentation," "object_pose," or other APIs that are explicitly forbidden.

Output Rules:
- You must output Python code in a single JSON blob with a "Final Answer" action after retrieving necessary information.
- Ensure your code is syntax-error-free by reviewing it carefully.

Follow these rules:
1. Always try the "VectorStoreSearch" tool first.
2. Only use "WebSearch" if the vector store does not contain the required information.
3. Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
4. Valid "action" values: "Final Answer" or {tool_names}
5. Provide only ONE action per JSON_BLOB
"""


# human prompt
human_prompt = """{input}  
{agent_scratchpad}  
(reminder to always respond in a JSON blob)"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

# tool render

prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

# create rag chain

chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

# create agent

agent_executor = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)
client = airsim.MultirotorClient()
client.confirmConnection()
camera_name = "0"
image_type = airsim.ImageType.Scene

while True:
    command = input("Enter your command (or !quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break

    response = agent_executor.invoke({"input": command})

    print("Gemini output:", response['output'])

    code = extract_python_code(response['output'])
    code_executor(code)