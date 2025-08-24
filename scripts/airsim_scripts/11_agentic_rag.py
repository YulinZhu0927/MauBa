import warnings
warnings.filterwarnings('ignore')

# set api key
import os
# from google.colab import userdata
os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

# load pdf
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"files/2307.05973v2.pdf")
documents = loader.load()

# split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

# load embedding model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})

# # create vectorstore
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)

# saving the vectorstore
vectorstore.save_local("vectorstore.db")

# create retriever
retriever = vectorstore.as_retriever()

# define web search
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=10)

# load llm
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")


# define vector search
from langchain.chains import RetrievalQA


def vector_search(query: str):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.invoke(query)


# define web search
def web_search(query: str):
    return web_search_tool.invoke(query)


# create tool call for vector search and web search
from langchain.tools import tool


@tool
def vector_search_tool(query: str) -> str:
    """Tool for searching the vector store."""
    return vector_search(query)


@tool
def web_search_tool_func(query: str) -> str:
    """Tool for performing web search."""
    return web_search(query)

# define tools for the agent
from langchain.agents import Tool
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

# define system prompt

system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
Always try the \"VectorStoreSearch\" tool first. Only use \"WebSearch\" if the vector store does not contain the required information.
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per 
JSON_BLOB"""

# human prompt
human_prompt = """{input}  
{agent_scratchpad}  
(reminder to always respond in a JSON blob)"""

# create prompt template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

# tool render
from langchain.tools.render import render_text_description_and_args
prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

# create rag chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

# create agent
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

print(agent_executor.invoke({"input": "Tell me about VoxPoser"}))
