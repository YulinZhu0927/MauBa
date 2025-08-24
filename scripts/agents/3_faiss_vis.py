import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

warnings.filterwarnings('ignore')

# 设置 API Key
os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

# ----------------------------
# 1. 加载 Python 文件，并生成 Document（包含元数据）
# ----------------------------
from langchain.schema import Document


def load_python_files_with_metadata(folder_path: str) -> list:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 将整个文件作为一个 Document，同时保存文件名作为 source
                doc = Document(page_content=content, metadata={"source": filename})
                documents.append(doc)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
    return documents


folder_path = r"F:\Tools\Python311\Lib\site-packages\airsim"
raw_documents = load_python_files_with_metadata(folder_path)
print(f"共加载到 {len(raw_documents)} 个原始文档。")

# ----------------------------
# 2. 拆分文档（例如每 500 字符一个片段）
# ----------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_documents = text_splitter.split_documents(raw_documents)
print(f"拆分后共获得 {len(split_documents)} 个文档片段。")

# ----------------------------
# 3. 构建 FAISS 向量库（使用拆分后的文档）
# ----------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# 如果之前已保存索引，可直接加载；否则构建新的：
# vectorstore = FAISS.from_documents(split_documents, embeddings)
# vectorstore.save_local("airsim_1.8.1_windows.db")
vectorstore = FAISS.load_local("airsim_1.8.1_windows.db", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
print("FAISS 向量库构建完成。")
print("FAISS ntotal:", vectorstore.index.ntotal)
print("Documents count:", len(split_documents))


# ----------------------------
# 4. 使用 PCA 可视化整个 RAG 结构，保存坐标到 DataFrame，并导出为 CSV 和 PDF
# ----------------------------
def visualize_rag_structure(vectorstore, documents):
    """
    使用 PCA 将 FAISS 向量库中的嵌入降到2维进行可视化。
    每个点代表一个文档片段，其颜色基于文档来源 (source) 分配。
    将降维后的坐标信息保存到 DataFrame 中，并导出为 CSV 和 PDF。
    """
    total_vectors = vectorstore.index.ntotal
    embeddings_arr = np.array([vectorstore.index.reconstruct(int(i)) for i in range(total_vectors)])

    # 根据 "source" 字段获取颜色，使用指定的四种颜色
    sources = [doc.metadata.get("source", "unknown") for doc in documents]
    unique_sources = list(set(sources))
    color_list = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF']
    color_mapping = {src: color_list[i % 4] for i, src in enumerate(unique_sources)}
    colors = [color_mapping[src] for src in sources]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings_arr)

    # 保存坐标信息到 DataFrame
    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["source"] = sources
    df.to_csv("rag_structure.csv", index=False)

    plt.figure(figsize=(12, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=src,
                          markerfacecolor=color_mapping[src], markersize=10) for src in unique_sources]
    plt.legend(handles=handles, title="Source", loc='best', fontsize=14, title_fontsize=16)
    plt.title("PCA Visualization of RAG Structure", fontsize=20)
    plt.xlabel("PCA Component 1", fontsize=16)
    plt.ylabel("PCA Component 2", fontsize=16)
    plt.tight_layout()
    plt.savefig("rag_structure.pdf")
    plt.show()

    return df


df_structure = visualize_rag_structure(vectorstore, split_documents)
print("RAG 结构坐标已保存到 df_structure。")
print(df_structure.head())


# ----------------------------
# 5. 使用 PCA 可视化 RAG 检索过程，链式箭头连接，并保存坐标到 DataFrame、导出为 CSV 和 PDF
# ----------------------------
def visualize_rag_retrieval(query, vectorstore, documents, embeddings, k=5):
    """
    对给定查询，通过 FAISS 检索 top-k 文档，然后使用 PCA 降维显示：
    - 所有 FAISS 向量以浅灰色显示（alpha=0.6）
    - 查询点以红色星号显示
    - 检索到的点以红色边框显示（无填充），并以链式箭头连接：
      query -> 第一个检索点 -> 第二个检索点 -> ...
    同时保存查询与检索点的坐标信息到 DataFrame 中，并导出为 CSV 和 PDF。
    """
    total_vectors = vectorstore.index.ntotal
    all_embeddings = np.array([vectorstore.index.reconstruct(int(i)) for i in range(total_vectors)])

    pca = PCA(n_components=2)
    reduced_all = pca.fit_transform(all_embeddings)

    # 生成查询嵌入并变换
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    query_coord = pca.transform(query_embedding)[0]

    # FAISS 检索 top-k（在原始向量空间中）
    D, I = vectorstore.index.search(query_embedding, k)
    retrieved_indices = [int(i) for i in I[0]]
    retrieved_coords = reduced_all[retrieved_indices]

    similarity_scores = np.exp(-D[0])
    scaler = MinMaxScaler()
    similarity_scores_norm = scaler.fit_transform(similarity_scores.reshape(-1, 1)).flatten()

    # 保存查询与检索点坐标到 DataFrame（链式顺序）
    rows = []
    rows.append({"type": "query", "order": 0, "label": "Query", "x": query_coord[0], "y": query_coord[1]})
    for order, idx in enumerate(retrieved_indices, start=1):
        doc = documents[idx]
        label = doc.metadata.get("function", doc.metadata.get("source", f"doc_{idx}"))
        # 注意：retrieved_coords 的顺序与 order 一致
        x, y = retrieved_coords[order - 1]
        rows.append({"type": "retrieved", "order": order, "label": label, "x": x, "y": y})
    df_retrieval = pd.DataFrame(rows, columns=["type", "order", "label", "x", "y"])
    df_retrieval.to_csv("rag_retrieval.csv", index=False)

    plt.figure(figsize=(12, 10))
    # 绘制所有点（浅灰色）
    plt.scatter(reduced_all[:, 0], reduced_all[:, 1], color="lightgray", alpha=0.6, label="All Points")
    # 绘制查询点（红色星号）
    plt.scatter(query_coord[0], query_coord[1], color="red", marker="*", s=200, label="Query")
    # 绘制链式箭头连接检索点
    chain_points = [query_coord] + list(retrieved_coords)
    for i in range(len(chain_points) - 1):
        start = chain_points[i]
        end = chain_points[i + 1]
        # 绘制箭头连接
        plt.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))
    # 绘制检索点（红色边框，无填充）
    for coord in retrieved_coords:
        plt.scatter(coord[0], coord[1], facecolors="none", edgecolors="red", s=150, alpha=0.6)

    plt.title("PCA Visualization of RAG Retrieval Process", fontsize=20)
    plt.xlabel("PCA Component 1", fontsize=16)
    plt.ylabel("PCA Component 2", fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("rag_retrieval.pdf")
    plt.show()

    return df_retrieval

# ----------------------------
# 6. 构建 RAG 交互代理（固定指令）
# ----------------------------
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description_and_args
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

web_search_tool = TavilySearchResults(k=10)


def vector_search(query: str):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.invoke(query)


def web_search(query: str):
    return web_search_tool.invoke(query)


@tool(description="Tool for searching the vector store for information.")
def vector_search_tool(query: str) -> str:
    return vector_search(query)


@tool(description="Tool for performing a web search for information.")
def web_search_tool_func(query: str) -> str:
    return web_search(query)


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

system_prompt = """
You are an intelligent assistant that helps me control the drone in AirSim and complete a series of tasks.
When I ask you to do something, you are supposed to give me Python code that is needed to achieve that task using AirSim and then an explanation of what that code does.
You are only allowed to use the airsim functions in version 1.8.1 under Windows OS.
You can use simple Python functions from libraries such as math, numpy and cv2.
Output the code in a single block and ensure it is syntactically correct.

Follow these rules:
1. Always try the "VectorStoreSearch" tool first.
2. Only use "WebSearch" if the vector store does not contain the required information.
3. Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
4. Valid "action" values: "Final Answer" or {tool_names}.
5. Provide only ONE action per JSON blob.
"""

human_prompt = """{input}  
{agent_scratchpad}  
(reminder to always respond in a JSON blob)"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)
prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp")

chain = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

# 固定指令
command = "After taking off to 10M, fly forward 50M, rotate clockwise 45 degrees and fly 10M, then land."
print("User Command:", command)

response = agent_executor.invoke({"input": command})
print("Gemini output:", response)


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


code = extract_python_code(response.get('output', ''))
if code is not None:
    print("Extracted code:\n", code)
    try:
        exec(code)
    except Exception as e:
        print("Error executing code:", e)

df_retrieval = visualize_rag_retrieval(command, vectorstore, split_documents, embeddings, k=5)
print("RAG 检索过程坐标已保存到 df_retrieval。")
print(df_retrieval)
