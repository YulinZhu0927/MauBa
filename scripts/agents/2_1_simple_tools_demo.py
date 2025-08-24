import warnings
warnings.filterwarnings('ignore')

import os
from PIL import Image

# 设置 API Key
os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
os.environ['TAVILY_API_KEY'] = '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'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

# 导入工具和模型
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import StructuredTool
from langchain.agents import Tool

# 定义联网搜索工具
web_search_tool = TavilySearchResults(k=10)

def web_search(query: str) -> str:
    """Perform a web search using Tavily."""
    return web_search_tool.invoke(query)

# 使用 StructuredTool 定义工具
web_search_tool_structured = StructuredTool.from_function(
    func=web_search,
    name="WebSearch",
    description="Use this tool to perform a web search for information."
)

tools = [web_search_tool_structured]

# 定义系统提示
system_prompt = """
You are an intelligent assistant that helps me solve problems by generating responses and explanations.
When I provide you with a textual description and an image, you must analyze them and produce an answer.
You have access to the following tool: {tool_names}.
Always respond in a JSON blob.
Follow these rules:
1. Use the WebSearch tool to look up any necessary information.
2. Provide your final answer in a JSON blob with action "Final Answer" and include your answer and explanation.
"""

# 定义人类提示
human_prompt = """{input}
{image_info}
{agent_scratchpad}
(Reminder: always respond in a JSON blob.)
"""

# 创建 prompt 模板
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
).partial(
    tool_names=", ".join([t.name for t in tools]),
    agent_scratchpad=MessagesPlaceholder(variable_name="agent_scratchpad")
)

# 创建链条
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

# 假设我们无法直接处理图像，这里将图像信息简化为文本描述
def preprocess_input(inputs):
    text_input = inputs["input"]
    image = inputs["image"]
    # 模拟图像分析：假设第四只猫是“橙色短毛猫”（实际需要图像识别模型）
    image_info = "图片中有多只猫咪，从左往右数第四只猫是橙色短毛猫。"
    return {"input": text_input, "image_info": image_info, "intermediate_steps": inputs.get("intermediate_steps", [])}

chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | preprocess_input
    | prompt
    | ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp", temperature=0.0)  # 使用支持多模态的模型
    | JSONAgentOutputParser()
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

# 测试输入
text_input = """图片中从左往右数第四只猫咪是什么品种？"""
image_path = r"test_img/cat.jpg"
image = Image.open(image_path)

response = agent_executor.invoke({"input": text_input, "image": image})
print("Gemini output: ", response)