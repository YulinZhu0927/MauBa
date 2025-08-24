import warnings

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.caches import InMemoryCache

warnings.filterwarnings('ignore')

from langchain.schema import Document
import airsim
import re

# set api key
import os
# from google.colab import userdata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description_and_args
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.tools import tool
from langchain.globals import set_llm_cache

# set_llm_cache(InMemoryCache())

global_namespace = globals()


def API_Key_init():
    os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
    os.environ['TAVILY_API_KEY'] = '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'
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

API_Key_init()

# load pdf
# from langchain_community.document_loaders import PyPDFLoader

# # 定义函数来读取Python文件内容
# def read_python_files(folder_path):
#     documents = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".py"):  # 确保只处理Python文件
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 content = file.read()
#                 documents.append(Document(page_content=content))
#     return documents
#
# # 指定包含Python文件的文件夹路径
# folder_path = r"F:\Tools\Python311\Lib\site-packages\airsim"
# documents = read_python_files(folder_path)
#
# # split documents
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# documents = text_splitter.split_documents(documents)
#
# # load embedding model


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                   encode_kwargs={"normalize_embeddings": True})
#
# # # create vectorstore

#
# vectorstore = FAISS.from_documents(documents, embeddings)
#
# # saving the vectorstore
# vectorstore.save_local("airsim_1.8.1_windows.db")

vectorstore = FAISS.load_local("airsim_1.8.1_windows.db", embeddings, allow_dangerous_deserialization=True)

# create retriever
retriever = vectorstore.as_retriever()

# define web search

web_search_tool = TavilySearchResults(k=10)

# load llm

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp")

# create tool call for vector search and web search
@tool
def vector_search_tool(query: str) -> str:
    """Tool for searching the vector store."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
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
            exec(code, global_namespace)
        except Exception as e:
            print("Error executing code:", e)

# define system prompt
system_prompt = """
You are an intelligent assistant that helps me control the drone in AirSim and complete a series of tasks.
When I ask you to do something, you are supposed to give me Python code that is needed to achieve that task using AirSim and then an explanation of what that code does.
You are only allowed to use the airsim functions in the version of 1.8.1 under the Windows OS, so you are not to use any other hypothetical functions that you think might exist.
You can use simple Python functions from libraries such as math, numpy and cv2.
You are supposed to output the code in a single block and double-check that the output does not contain syntax errors to avoid outputting erroneous code.
If the instruction contains a situation that needs to be located, you should ask me first instead of trying to use the AirSim API for detecting scene objects.

Follow these rules:
    1. Always try the \"VectorStoreSearch\" tool first.
    2. Only use \"WebSearch\" if the vector store does not contain the required information.
    3. Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
    4. Valid "action" values: "Final Answer" or {tool_names}
    5. Provide only ONE action per JSON_BLOB
"""

# human prompt
human_prompt = """{input}  
{agent_scratchpad}  
(reminder to always respond in a JSON blob)"""

# create prompt template

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

# output_parser = StructuredOutputParser.from_response_schemas([
#     ResponseSchema(name="input", description="Instructions entered by the user"),
#     ResponseSchema(name="code", description="Code generated from the instructions"),
#     ResponseSchema(name="explanation", description="A detailed description of the generated code"),
# ])

while True:
    command = input("Enter your command (or !quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break

    response = agent_executor.invoke({"input": command})

    print("Gemini output:", response['output'])
    # print("Whole response", response)

    code = extract_python_code(response['output'])
    code_executor(code)
