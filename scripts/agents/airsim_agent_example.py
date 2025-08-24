import warnings
warnings.filterwarnings('ignore')

import os
from typing import TypedDict, Optional, List, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 导入我们的AirSimAgent类
from airsim_agent import AirSimAgent

# 定义消息状态
class MessagesState(TypedDict):
    messages: Annotated[List[Any], add_messages]  # 使用 add_messages 管理消息列表
    next: Optional[str]                           # 下一个节点

# 初始化AirSimAgent
agent = AirSimAgent(verbose=True)

# 定义节点函数
def airsim_node(state: MessagesState) -> MessagesState:
    """处理AirSim相关的命令"""
    # 获取最后一条消息内容
    last_message = state["messages"][-1].content
    
    # 调用AirSimAgent处理命令
    response = agent.invoke(last_message)
    
    # 提取并执行代码
    code = agent.extract_python_code(response['output'])
    agent.code_executor(code)
    
    # 返回结果
    return {"messages": [{"role": "assistant", "content": response['output']}], "next": "end"}

# 创建工作流图
def create_graph():
    # 创建状态图
    workflow = StateGraph(MessagesState)
    
    # 添加节点
    workflow.add_node("airsim_node", airsim_node)
    
    # 设置入口节点
    workflow.set_entry_point("airsim_node")
    
    # 设置边缘连接
    workflow.add_edge("airsim_node", END)
    
    # 编译图
    return workflow.compile()

# 创建应用
graph = create_graph()

# 示例：如何在应用中使用
def run_example():
    # 初始化状态
    state = {"messages": [{"role": "user", "content": "让无人机起飞并向前飞行10米"}]}
    
    # 运行图
    for s in graph.stream(state):
        if "messages" in s:
            for message in s["messages"]:
                if message["role"] == "assistant":
                    print(f"Assistant: {message['content']}")

# 交互式运行
def run_interactive():
    print("AirSim Agent LangGraph 示例 (输入 '!quit' 退出)")
    
    while True:
        user_input = input("\n请输入命令: ")
        if user_input.lower() in ["!quit", "!exit"]:
            break
            
        # 初始化状态
        state = {"messages": [{"role": "user", "content": user_input}]}
        
        # 运行图
        for s in graph.stream(state):
            if "messages" in s:
                for message in s["messages"]:
                    if message["role"] == "assistant":
                        print(f"\nAssistant: {message['content']}")

if __name__ == "__main__":
    # 运行交互式示例
    run_interactive()