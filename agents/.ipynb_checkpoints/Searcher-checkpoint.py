## lack the momery component
## add some information -->  print("Begin the xxxxxx  task") & print("Let's begin xxx tool usage.")

import os
from dotenv import load_dotenv     # pip install python-dotenv
import torch
import requests
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

class SearcherAgent:
    def __init__(self, question: str):
        load_dotenv()
        self.setup_environment()
        self.tools = self.initialize_tools()
        self.llm = ChatOpenAI(
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        self.model_with_tools = self.llm.bind_tools(self.tools)
        self.memory = []
        self.question = question
        self.summarizer = self._initialize_summarizer()
    def setup_environment(self):
        
        # openai
        os.environ["OPENAI_API_TYPE"] = "open_ai"
        os.environ["OPENAI_API_KEY"] = "sk-dfrFMkQ4O7U9t8tD7dD665854eC04c70900dD4Ea79732029"
        os.environ["OPENAI_API_BASE"] = "https://api.bltcy.ai/v1"
        # langchain
        os.environ["LANGCHAIN_API_KEY"] = "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"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
        os.environ["LANGCHAIN_PROJECT"] = "This is used for my search agent in toursim MAS"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # asknews
        os.environ["ASKNEWS_CLIENT_ID"] = "your asknews client id"
        os.environ["ASKNEWS_CLIENT_SECRET"] = "your asknews client secret"
        # open weather app
        os.environ["OPENWEATHERMAP_API_KEY"] = "your weather api key"

        # bing
        os.environ["BING_SUBSCRIPTION_KEY"] = "<key>"
        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"


    def initialize_tools(self, max_results: int = 2, language: str = "en") -> list:
        tools = []
        # Initialize Wikipedia Tool
        
        try:
            wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
            wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
        except Exception as e:
            print("Wikipedia Tool cannot be initialized:", e)

        # Initialize Arxiv Tool
        try:
            arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
            arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
            tools.append(arxiv_tool)
        except Exception as e:
            print("Arxiv Tool cannot be initialized:", e)

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
                    api_wrapper_bing = BingSearchAPIWrapper(api_key=bing_key, search_url=bing_url,k = max_results)
                    bing_search_tool = BingSearchResults(api_wrapper=api_wrapper_bing)
                    tools.append(bing_search_tool)
                else:
                    print("Bing Search Tool cannot be initialized: Invalid API key or URL.")
            else:
                print("Bing Search Tool cannot be initialized: BING_SUBSCRIPTION_KEY or BING_SEARCH_URL not found.")
        except Exception as e:
            print("Bing Search Tool cannot be initialized:", e)

        # Initialize YouTube Search Tool

        # pip install --upgrade --quiet  youtube_search
        try:
            youtube_tool = YouTubeSearchTool()
            tools.append(youtube_tool)
            # use --->   youtube_tool.run("lex friedman, {}".format(max_results))
        except Exception as e:
            print("YouTube Tool cannot be initialized:", e)

        # Ionic Tool

        # pip install langchain langchain_openai langchainhub
        # pip install ionic-langchain
        try:
            ionic_tool = IonicTool().tool()
            tools.append(ionic_tool)
        except Exception as e:
            print("Ionic Tool cannot be initialized:", e)

        # Weather App

        # pip install pyowm
        try:
            if "OPENWEATHERMAP_API_KEY" in os.environ:
                weather = OpenWeatherMapAPIWrapper()
                # use  -->   weather_data = weather.run("London,GB")
                tools.append(weather)
            else:
                print("Weather Tool cannot be initialized: OPENWEATHERMAP_API_KEY not found.")
        except Exception as e:
            print("Weather Tool cannot be initialized:", e)

        # Initialize AskNews Search Tool

        # pip install -U langchain-community asknews
        try:
            asknews_tool = AskNewsSearch(max_results)
            tools.append(asknews_tool)
        except Exception as e:
            print("AskNews Tool cannot be initialized:", e)

        return tools
    

    def get_response(self) -> str:
        system_prompt = self.question
        prompt = "Hi"
        messages = [
            SystemMessage(content = system_prompt),
            HumanMessage(content = prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def get_tool_calls(self) -> str:
        system_prompt = self.question
        prompt = "Hi"
        messages = [
            SystemMessage(content = system_prompt),
            HumanMessage(content = prompt),
        ]
        response = self.model_with_tools.invoke(messages)
        return response.tool_calls
    
    def _initialize_summarizer(self) -> None:
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

    def get_summary(self, text: str, max_length: int = 150, min_length: int = 30):
        return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    
    def process_text(self, text: str, summary_required: bool = True) -> str:  # 可以将required丢到init
        if summary_required:
            summarized_text = self.get_summary(text)
            return summarized_text
        return text

    def create_agent_executor(self) -> str:
        ## Final Output + Summerize
        prompt_content = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(self.llm, self.tools, prompt_content)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        ask_question = self.question
        raw_response = agent_executor.invoke({"input":ask_question})
        
        summarized_response = self.process_text(raw_response)
        
        return summarized_response
"""

if __name__ == "__main__":
    question = ""
    searcher_agent = SearcherAgent("Could you tell me some information about tourism about qingyuan city, Guangdong Province, Chain.") 
    # without tool --> pure llm output
    response = searcher_agent.get_response()
    print("Response:", response)
    # with the tool usage  
    tool_calls = searcher_agent.get_tool_calls()
    print("Tool Calls:", tool_calls)

    agent_executor = searcher_agent.create_agent_executor()

"""