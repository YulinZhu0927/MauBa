import warnings
from typing import Dict, Any, Optional, List

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

warnings.filterwarnings('ignore')

from langchain.schema import Document
import airsim
import re
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description_and_args
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.tools import tool


class CoderAgent:
    """
    AirSim Agent class for controlling drones in AirSim using LLM.
    This class encapsulates the functionality to interact with AirSim using LangChain agents.
    """
    
    def __init__(self, 
                 vectorstore_path: str = "airsim_1.8.1_windows.db",
                 model_name: str = "gemini-2.0-pro-exp",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 init_api_keys: bool = True,
                 verbose: bool = True):
        """
        Initialize the AirSim Agent.
        
        Args:
            vectorstore_path: Path to the FAISS vector store
            model_name: Name of the LLM model to use
            embedding_model: Name of the embedding model to use
            init_api_keys: Whether to initialize API keys
            verbose: Whether to enable verbose mode for the agent
        """
        self.verbose = verbose
        
        # Initialize API keys if requested
        if init_api_keys:
            self._init_api_keys()
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load vector store
        self.vectorstore = FAISS.load_local(
            vectorstore_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever()
        
        # Initialize web search tool
        self.web_search_tool = TavilySearchResults(k=10)
        
        # Load LLM
        self.llm = ChatGoogleGenerativeAI(model=model_name)

        self.output_parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="input", description="Instructions entered by the user"),
            ResponseSchema(name="code", description="Code generated from the instructions"),
            ResponseSchema(name="explanation", description="A detailed description of the generated code"),
        ])
        
        # Initialize AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.camera_name = "0"
        self.image_type = airsim.ImageType.Scene
        
        # Create tools and agent
        self._create_tools()
        self._create_agent()
    
    def _init_api_keys(self):
        """
        Initialize API keys for various services.
        """
        os.environ['ATHINA_API_KEY'] = 'zeEbaU9m_zpNSDVKsbXa69gq5K-BUAcP'
        os.environ['TAVILY_API_KEY'] = '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'
        os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88'

    @tool
    def _vector_search_tool(self, query: str) -> str:
        """
        Tool for searching the vector store.
        
        Args:
            query: The query to search for
            
        Returns:
            The search results
        """
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)
        return qa_chain.invoke(query)

    @tool
    def _web_search_tool_func(self, query: str) -> str:
        """
        Tool for performing web search.
        
        Args:
            query: The query to search for
            
        Returns:
            The search results
        """
        return self.web_search_tool.invoke(query)
    
    def _create_tools(self):
        """
        Create tools for the agent.
        """
        self.tools = [
            Tool(
                name="VectorStoreSearch",
                func=self._vector_search_tool,
                description="Use this to search the vector store for information."
            ),
            Tool(
                name="WebSearch",
                func=self._web_search_tool_func,
                description="Use this to perform a web search for information."
            ),
        ]
    
    def _create_agent(self):
        """
        Create the agent executor.
        """
        # Define system prompt
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

        # Human prompt
        human_prompt = """{input}  
        {agent_scratchpad}  
        (reminder to always respond in a JSON blob)"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )

        # Tool render
        prompt = prompt.partial(
            tools=render_text_description_and_args(list(self.tools)),
            tool_names=", ".join([t.name for t in self.tools]),
        )

        # Create RAG chain
        chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | self.llm
            | JSONAgentOutputParser()
        )

        # Create agent
        self.agent_executor = AgentExecutor(
            agent=chain,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=self.verbose
        )
    
    def extract_python_code(self, content: str) -> Optional[str]:
        """
        Extract Python code from content.
        
        Args:
            content: The content to extract code from
            
        Returns:
            The extracted code or None if no code is found
        """
        code_block_regex = re.compile(r"```(.*)```", re.DOTALL)
        code_blocks = code_block_regex.findall(content)

        if code_blocks:
            full_code = "\n".join(code_blocks)

            if full_code.startswith("python"):
                full_code = full_code[7:]

            return full_code
        else:
            return None
    
    def code_executor(self, code: Optional[str]) -> None:
        """
        Execute Python code.
        
        Args:
            code: The code to execute
        """
        if code is not None:
            print("Extracted code:\n")
            print(code)
            try:
                exec(code)
            except Exception as e:
                print("Error executing code:", e)

    def invoke(self, command: str) -> Dict[str, Any]:
        """
        Invoke the agent with a command.

        Args:
            command: The command to invoke the agent with

        Returns:
            The response from the agent
        """
        response = self.agent_executor.invoke({"input": command})
        return response
    
    def run_interactive(self) -> None:
        """
        Run the agent in interactive mode.
        """
        while True:
            command = input("Enter your command (or !quit to exit): ")
            if command.lower() in ["!quit", "!exit"]:
                break

            response = self.invoke(command)

            print("Gemini output:", response['output'])

            code = self.extract_python_code(response['output'])
            self.code_executor(code)
    
    def run_command(self, command: str) -> Dict[str, Any]:
        """
        Run a single command and execute the code.
        
        Args:
            command: The command to run
            
        Returns:
            The response from the agent
        """
        response = self.invoke(command)
        code = self.extract_python_code(response['output'])
        self.code_executor(code)
        return response


# Example usage
if __name__ == "__main__":
    agent = CoderAgent()
    # agent.run_interactive()
    agent.run_command("向前飞行10M，注意状态检查")