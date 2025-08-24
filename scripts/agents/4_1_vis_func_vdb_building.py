import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                   encode_kwargs={"normalize_embeddings": True})

# 定义函数来读取Python文件内容
def read_python_files(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        documents.append(Document(page_content=content))
    return documents

# 指定包含Python文件的文件夹路径
file_path = r"../object_detection/gemini_od/2_1_gemini_dino_integration_testing.py"
documents = read_python_files(file_path)

# split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("airsim_vision.db")