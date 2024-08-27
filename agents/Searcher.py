import os
import json
import torch
import warnings
import requests
from dotenv import load_dotenv
from transformers import pipeline
from typing import List, Dict, Union, Any, Tuple, Optional
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.tools.bing_search.tool import BingSearchResults
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, OpenWeatherMapAPIWrapper
from langchain_community.tools.asknews import AskNewsSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_openai_tools_agent, create_tool_calling_agent, AgentExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from ionic_langchain.tool import Ionic, IonicTool
from transformers import pipeline, AutoTokenizer
from transformers.pipelines import SummarizationPipeline
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


class SearcherAgent:

    """
    Initializes the SearcherAgent class with a question, language, and maximum results for tool calls.

    Parameters:
    - question (str): The user's question or query.
    - language (str): The language to be used for summarization and embedding (default is "en").
    - max_results (int): Maximum number of results to return from tools (default is 2).
    """
    def __init__(self, question: str, language: str, max_results: int, key: str, 
                 base: str, lang_key: str, lang_pro: str) -> None:
        
        self.question = question
        self.language = language
        self.max_results = max_results
        self.memory = []
        
        self.key = key
        self.base = base
        self.lang_key = lang_key
        self.lang_pro = lang_pro

        load_dotenv()
        self.setup_environment()
        self.tools = self.initialize_tools(max_results=self.max_results, language=self.language)

        self.llm = self._initialize_llm()
        self.model_with_tools = self.llm.bind_tools(self.tools)
        self.summarizer = self._initialize_summarizer()

    def setup_environment(self) -> None:
        try:
            os.environ["OPENAI_API_TYPE"] = "open_ai"
            os.environ["OPENAI_API_KEY"] = self.key
            os.environ["OPENAI_API_BASE"] = self.base
            os.environ["LANGCHAIN_API_KEY"] = self.lang_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
            os.environ["LANGCHAIN_PROJECT"] = self.lang_pro
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # if you don't have these TOOL API KEYs, you can also don't choose them.
            if "BING_SUBSCRIPTION_KEY" not in os.environ or "BING_SEARCH_URL" not in os.environ:
                print("Bing Search Tool cannot be initialized: BING_SUBSCRIPTION_KEY or BING_SEARCH_URL not found.")
            if "OPENWEATHERMAP_API_KEY" not in os.environ:
                print("Weather Tool cannot be initialized: OPENWEATHERMAP_API_KEY not found.")
            if "ASKNEWS_CLIENT_ID" not in os.environ or "ASKNEWS_CLIENT_SECRET" not in os.environ:
                print("AskNews Tool cannot be initialized: ASKNEWS_CLIENT_ID or ASKNEWS_CLIENT_SECRET not found.")
        except Exception as e:
            print(f"Error setting up environment variables: {e}")

    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initializes the LLM (Language Model) used by the agent.

        Returns:
        - ChatOpenAI: The initialized LLM model.
        """
        try:
            return ChatOpenAI(
                openai_api_base = self.base,
                openai_api_key = self.key
            )
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            raise

    def initialize_tools(self, max_results: int = 2, language: str = "en") -> List:
        """
        Initializes various tools for information retrieval, search, and summarization.

        Parameters:
        - max_results (int): Maximum number of results to return from tools.
        - language (str): The language to be used for embeddings (default is "en").

        Returns:
        - List: A list of initialized tools.
        """
        tools = []

        # Initialize Wikipedia Tool
        try:
            wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=self.max_results, doc_content_chars_max=200)
            wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
            tools.append(wiki_tool)
        except Exception as e:
            print("Wikipedia Tool cannot be initialized:", e)

        # Load Documents and Initialize Vectorstore
        try:
            loader = WebBaseLoader("https://docs.smith.langchain.com/")
            docs = loader.load()
            documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

            try:
                if language == "en":
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                elif language == "chinese":
                    embeddings = HuggingFaceEmbeddings(model_name="uer/sbert-base-chinese-nli")
                else:
                    raise ValueError("Unsupported language. Please choose 'en' or 'chinese'.")
            except Exception as e:
                print("HuggingFace embeddings could not be loaded, falling back to GPT4AllEmbeddings:", e)
                model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
                gpt4all_kwargs = {'allow_download': 'True'}
                embeddings = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)

            vectordb = FAISS.from_documents(documents, embedding=embeddings)
            retriever = vectordb.as_retriever()
            retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about LangSmith.")
            tools.append(retriever_tool)
        except Exception as e:
            print("Retriever Tool cannot be initialized:", e)

        # Initialize Bing Search Tool
        try:
            if "BING_SUBSCRIPTION_KEY" in os.environ and "BING_SEARCH_URL" in os.environ:
                bing_key = os.environ["BING_SUBSCRIPTION_KEY"]
                bing_url = os.environ["BING_SEARCH_URL"]
                response = requests.get(bing_url, headers={"Ocp-Apim-Subscription-Key": bing_key}, params={"q": "test"})
                if response.status_code == 200:
                    api_wrapper_bing = BingSearchAPIWrapper(api_key=bing_key, search_url=bing_url, k=max_results)
                    bing_search_tool = BingSearchResults(api_wrapper=api_wrapper_bing)
                    tools.append(bing_search_tool)
                else:
                    print("Bing Search Tool cannot be initialized: Invalid API key or URL.")
            else:
                print("Bing Search Tool cannot be initialized: BING_SUBSCRIPTION_KEY or BING_SEARCH_URL not found.")
        except Exception as e:
            print("Bing Search Tool cannot be initialized:", e)

        # Initialize Ionic Tool
        try:
            ionic_tool = IonicTool().tool()
            tools.append(ionic_tool)
        except Exception as e:
            print("Ionic Tool cannot be initialized:", e)

        # Initialize Weather Tool
        
        try:
            if "OPENWEATHERMAP_API_KEY" in os.environ:
                weather = OpenWeatherMapAPIWrapper()
                tools.append(weather)
            else:
                print("Weather Tool cannot be initialized: OPENWEATHERMAP_API_KEY not found.")
        except Exception as e:
            print("Weather Tool cannot be initialized:", e)

        # Initialize AskNews Search Tool
        try:
            asknews_tool = AskNewsSearch(max_results)
            tools.append(asknews_tool)
        except Exception as e:
            print("AskNews Tool cannot be initialized:", e)

        return tools
    
    def get_response(self) -> str:
        """
        Gets a basic response from the LLM based on the user's question.

        Returns:
        - str: The response content from the LLM.
        """
        try:
            system_prompt = self.question
            prompt = "Hi"
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Failed to get response: {e}")
            return ""

    def get_tool_calls(self) -> str:
        """
        Gets tool calls from the LLM when interacting with the initialized tools.

        Returns:
        - str: The tool call responses from the LLM.
        """
        try:
            system_prompt = self.question
            prompt = "Hi"
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
            response = self.model_with_tools.invoke(messages)
            return response.tool_calls
        except Exception as e:
            print(f"Failed to get tool calls: {e}")
            return ""
    
    def _initialize_summarizer(self) -> Optional[pipeline]:
        try:
            if self.language == "en":
                model_name = "facebook/bart-large-cnn"
            elif self.language == "chinese":
                model_name = "uer/bart-large-chinese-cluecorpussummary"
            else:
                raise ValueError("Unsupported language. Please choose 'en' or 'chinese'.")

            if torch.cuda.is_available():
                return pipeline("summarization", model=model_name, device=0)
            else:
                return pipeline("summarization", model=model_name)
        except Exception as e:
            print(f"Failed to initialize summarizer: {e}")
            return None  # Gracefully handle failure

    def get_summary(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """
        Summarizes the given text using the initialized summarizer.

        Parameters:
        - text (str): The text to be summarized.
        - max_length (int): Maximum length of the summary (default is 150).
        - min_length (int): Minimum length of the summary (default is 30).

        Returns:
        - str: The summarized text.
        """
        try:
            return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Failed to generate summary: {e}")
            return text
    
    def process_text(self, text: str, summary_required: bool = True) -> str:
        """
        Processes the text, optionally summarizing it.

        Parameters:
        - text (str): The text to process.
        - summary_required (bool): Whether summarization is required (default is True).

        Returns:
        - str: The processed text.
        """
        try:
            if summary_required and self.summarizer:
                return self.get_summary(text)
            return text
        except Exception as e:
            print(f"Failed to process text: {e}")
            return text

    def create_agent_executor(self) -> str:
        """
        Creates an agent executor to handle tool-based queries, processes the response, and optionally summarizes it.

        Returns:
        - str: The final processed response.
        """
        try:
            prompt_content = hub.pull("hwchase17/openai-functions-agent")
            agent = create_tool_calling_agent(self.llm, self.tools, prompt_content)
            agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            
            raw_response = agent_executor.invoke({"input": self.question})
            summarized_response = self.process_text(raw_response)

            return summarized_response
        except Exception as e:
            print(f"Failed to create agent executor: {e}")
            return ""