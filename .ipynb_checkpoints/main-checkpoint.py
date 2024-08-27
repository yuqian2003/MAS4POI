import os
import time
import random
import logging
import argparse
from agents.Data_Agent import DataAgent
from agents.ItemAgent import ItemAgent
from agents.Manager import ManageAgent
from agents.Reflector import ReflectorAgent, ReflectionStrategy
# from agents.Searcher import SearcherAgent      
from agents.Navigator import NaviAgent
from datetime import datetime

"""
DataAgent  --> 输出数据 --> 2. ManageAgent  --> 分配任务 --> 3. ItemAgent  --> 返回推荐结果 --> 4. ManageAgent  
--> 汇总结果 --> 5. ReflectorAgent  --> 输出最终结果
"""
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
"""
def main():
    parser = argparse.ArgumentParser(description='Multi Agent system for Next POI Recommendation.')

    parser.add_argument('--datasetName', type=str, choices=['nyc', 'tky', 'ca'], default='nyc', help='nyc/tky/ca')
    parser.add_argument('--case_num', type=int, default=100, help='Number of cases to process')
    parser.add_argument('--filePath', type=str, default='./data/nyc/raw/NYC_{}.csv',help='This is the dataset path')
    parser.add_argument('--seed', type=int, default=2070, help='seed for reproducibility')
    parser.add_argument('--task', type=str, choices=['poi', 'navigator', 'search'], default='poi', help='sub task')
    parser.add_argument('--stratege', type=str, choices=['REFLECTION','LAST_ATTEMPT','LAST_ATTEMPT_AND_REFLECTION','NONE'],
                        default='REFLECTION', help='This is the reflect stratege.')
    # log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to log directory')
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    # parser.add_argument('--repeat', type=int, default=5, help='number of repeats with seeds [seed, seed+repeat)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f'Arguments received: datasetName={args.datasetName}, case_num={args.case_num}')

    # initial
    data_agent = DataAgent(filePath=args.filePath, datasetName=args.datasetName, case_num=args.case_num)
    item_agent = ItemAgent(llm=args.LLM, logger=logger, temperature=args.temperature)

    print("Begining the Workflow")
    start_time = time.time()
    workflow = ManageAgent(data_agent, item_agent,
                            None, None, args.task, args.seed, args.case_num)
    end_time = time.time()
    total_time = end_time - start_time
    
    # metric
    results_dir = './POI_results/'
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f'Ensured results directory exists: {results_dir}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    resultPath = '{}/POI_workflow_{}_{}_{}.txt'.format(results_dir, args.datasetName, args.seed, timestamp)

    # Run the workflow and save results
    acc1, acc5, acc10, mrr = workflow.run_workflow()
    with open(resultPath, 'w') as file:
        file.write(f'ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}')
        logger.info(f'Results written to {resultPath}')
    logger.info(f'Results saved to {resultPath}')
"""
def main():
    parser = argparse.ArgumentParser(description='Multi Agent system for tourism.')

    parser.add_argument('--LLM', type=str, choices=['gpt, claude, spark, qwen, ernie, glm'],
                         default='gpt', help='The LLM models we use.')
    parser.add_argument('--temperature', type=float, default=0., help='The LLM temperature.')
    parser.add_argument('--datasetName', type=str, choices=['nyc', 'tky', 'ca'], default='nyc', help='nyc/tky/ca')
    parser.add_argument('--case_num', type=int, default=100, help='Number of cases to process')
    parser.add_argument('--filePath', type=str, default='./data/nyc/raw/NYC_{}.csv',help='This is the dataset path')
    parser.add_argument('--seed', type=int, default=2070, help='seed for reproducibility')
    parser.add_argument('--task', type=str, choices=['poi', 'navigator', 'search'], default='poi', help='sub task')
    parser.add_argument('--stratege', type=str, choices=['REFLECTION','LAST_ATTEMPT','LAST_ATTEMPT_AND_REFLECTION','NONE'],
                        default='REFLECTION', help='This is the reflect stratege.')

    # log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to dataset')
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{args.datasetName}_{timestamp}_workflow.log')
    logger = setup_logger(log_file)
    logger.info(f'Arguments received: datasetName={args.datasetName}, case_num={args.case_num}')

    # initial
    data_agent = DataAgent(filePath=args.filePath, datasetName=args.datasetName, case_num=args.case_num)
    item_agent = ItemAgent(llm=args.LLM, logger=logger, temperature=args.temperature)
    search_agent = None
    navigator_agent = None

    logger.info("Beginning the Workflow")
    start_time = time.time()
    workflow = ManageAgent(data_agent, item_agent, search_agent, navigator_agent,
                           args.task, args.seed, args.case_num, logger, llm=args.LLM)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info("Total time cost is: {}".format(total_time))

    # metric
    #results_dir = './POI_results/'
    #os.makedirs(results_dir, exist_ok=True)
    #logger.info(f'Ensured results directory exists: {results_dir}')
    #resultPath = f'{results_dir}/POI_workflow_{args.datasetName}_{args.seed}_{timestamp}.txt'
    base_dir = './POI_results/'
    results_dir = os.path.join(base_dir, args.datasetName)
    os.makedirs(results_dir, exist_ok=True)
    resultPath = f'{results_dir}/POI_workflow_{args.LLM}_{args.seed}_{timestamp}.txt'
    # Run the workflow and save results
    acc1, acc5, acc10, mrr = workflow.run_workflow()
    with open(resultPath, 'w') as file:
        file.write(f'ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}')
        logger.info(f'Results written to {resultPath}')
    logger.info(f'Results saved to {resultPath}')


if __name__ == '__main__':
    main()