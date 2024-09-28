import os
import time
import random
import logging
import argparse
from datetime import datetime
from agents.Data_Agent import DataAgent
from agents.Analyst import Analyst
from agents.Manager import ManageAgent
from agents.Reflector import ReflectorAgent
from agents.Searcher import SearcherAgent      
from agents.Navigator import NaviAgent


def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main():
    parser = argparse.ArgumentParser(description='MAS4POI: a Multi Agent Collaboration system for Next POI Recommendation.')
    
    # Initial
    parser.add_argument('--key', type=str, default = "YOUR_OPENAI_API_KEY", 
                help='API key for OpenAI or other LLM providers. This key is used to authenticate requests to the LLM service.')
    
    parser.add_argument('--base', type=str, default = "YOUR_OPENAI_BASE", 
                help='Base URL for the API client. Modify this if you are using a custom LLM endpoint instead of OpenAI directly.')
    
    parser.add_argument('--LLM', type=str, choices=['gpt, gemini, moonshot, qwen, claude'],
                default='gpt', help='Select the LLM model to use. Options include GPT, Gemini, Moonshot, Qwen, and Claude.')
    
    parser.add_argument('--temperature', type=float, default=0., 
            help='Temperature setting for the LLM. Controls the randomness of the model\'s responses. Lower values make the output more deterministic.')
    
    parser.add_argument('--seed', type=int, default=4090, 
                    help='Random seed for reproducibility. This ensures that results are consistent across runs.')
    
    parser.add_argument('--task', type=str, choices=['poi', 'navigator', 'search'], default='poi', 
            help='The task to execute. Options include "poi" for Point of Interest recommendation, "navigator" for route planning, and "search" for information retrieval.')
    
    # Reflector
    parser.add_argument('--times',type=int, default=1, 
        help='Number of reflection attempts for error correction. If initial predictions are incorrect, Reflector will try up to this many times to improve the result.')

    # Navigator 
    parser.add_argument('--amap_api_key', type=str, default ="YOUR_AMAP_API", 
                        help='API key for Amap (Gaode Map) services, used for retrieving geolocation and route information.')
    
    parser.add_argument('--use_coordinates',type=bool, default=True, 
            help='Boolean flag indicating whether the navigator input should use coordinates (True) or address strings (False).')
    
    parser.add_argument('--source',type=str, default="113.52,22.35", 
                        help='Source location as "longitude,latitude" (e.g., "113.52,22.35")')
    
    parser.add_argument('--target',type=str, default="113.59,22.35", 
                        help='Target location as "longitude,latitude" (e.g., "120.44,36.22")')
    
    parser.add_argument('--city',type=str, default="珠海",
            help='The city in which navigation will take place. Used to refine geolocation services.')
    
    # Data Agent
    parser.add_argument('--group', type=str, choices=['very_active','normal','inactive','overall'],
                    default='overall', 
                    help='User activity group for data processing in DataAgent. Choose "very_active", "normal", "inactive", or "overall".')
    
    parser.add_argument('--datasetName', type=str, choices=['nyc', 'tky'], default='nyc', 
                         help='Dataset name to use for processing. Options include "nyc" (New York City) and "tky" (Tokyo).')
    
    parser.add_argument('--case_num', type=int, default=100, 
                help='Number of cases to process in the workflow. Determines how many test cases will be evaluated.')
    
    parser.add_argument('--filePath', type=str, default='./data/nyc/raw/NYC_{}.csv',
                help='Path to the dataset files. This should be the directory containing the raw data files.')

    # Search Agent
    parser.add_argument('--language', type=str, default='en', choices = ['en','chinese'], 
                help='Language to use for search and summarization tasks. Supported options are "en" (English) and "chinese".')
    
    parser.add_argument('--question', type=str, default='What is Multi-Agent system?',
                help='The question or query to be processed by the Search Agent. For example, a user could ask about Multi-Agent systems.')
    
    parser.add_argument('--max_results', type=int, default=2,
                help='Maximum number of results to retrieve from the search tools (e.g., Wikipedia, Bing). Defines the depth of interaction with the search tools.')
    
    parser.add_argument('--LANGCHAIN_API_KEY', type=str, 
                default="YOUR_LANGCHAIN_API_KEY", 
                help='API key for LangChain, used for interacting with LangChain-powered services and integrations.')
    
    parser.add_argument('--LANGCHAIN_PROJECT', type=str, 
                        default="Search Agent for Tourism MAS",
                        help='Project name for LangChain API tracking. Helps to organize and monitor API usage by project.')

    # log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', 
                help='Directory path for saving log files. This is where workflow logs will be stored.')
    
    parser.add_argument('--report_freq', type=int, default=30, 
                help='Frequency for reporting progress during execution, measured in seconds.')
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, args.datasetName, args.group, str(args.case_num))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.LLM}_{args.seed}_workflow.log')
    logger = setup_logger(log_file)
    # logger.info(f'Arguments received: datasetName={args.datasetName}, case_num={args.case_num}')

    # The Agent Initial
    data_agent = DataAgent(filePath=args.filePath, datasetName=args.datasetName, case_num=args.case_num, group=args.group)
        analyst = Analyst(llm=args.LLM, 
                          logger=logger, 
                          temperature=args.temperature, 
                          key = args.key, base = args.base, 
                          self.prompt_id = args.prompt_id)
    if args.task == 'search':
        search_agent = SearcherAgent(
                question=args.question,
                language=args.language,
                max_results=args.max_results,
                key=args.key, 
                base=args.base,
                lang_key = args.LANGCHAIN_API_KEY,
                lang_pro = args.LANGCHAIN_PROJECT
            )
    else:
        search_agent = None
    if args.task == 'navigator':
        navi_agent = NaviAgent(amap_api_key=args.amap_api_key , source=args.source, target=args.target, 
                                   city=args.city, llm=args.LLM, logger=logger, temperature=args.temperature,
                                   key = args.key, base = args.base, 
                                   use_coordinates = args.use_coordinates)
    else:
        navi_agent = None
    logger.info("Beginning the Workflow")
    start_time = time.time()
    workflow = ManageAgent(data_agent, analyst, search_agent, navi_agent,
                           args.task, args.seed, args.case_num, logger, llm=args.LLM, 
                           times = args.times, datasetName=args.datasetName,
                           amap_api_key=args.amap_api_key, city = args.city, 
                           source = args.source, target = args.target, 
                           key = args.key, base = args.base,
                           use_coordinates = args.use_coordinates
                           )

    base_dir = './POI_results/'
    results_dir = os.path.join(base_dir, args.datasetName)
    os.makedirs(results_dir, exist_ok=True)
    resultPath = f'{results_dir}/POI_workflow_{args.LLM}_{args.seed}_{timestamp}.txt'
    # Run the workflow and save results
    
    if args.task == 'poi':
        acc1, acc5, acc10, mrr = workflow.run_poi_workflow()

        with open(resultPath, 'w') as file:
            file.write(f'ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}')
            logger.info(f'Results written to {resultPath}')
        logger.info(f'Results saved to {resultPath}')
    
    elif args.task =='search':
        workflow.run_search_workflow()
    
    else:
        workflow.run_navigator_workflow()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info("Total time cost is: {}".format(total_time))

if __name__ == '__main__':
    main()
