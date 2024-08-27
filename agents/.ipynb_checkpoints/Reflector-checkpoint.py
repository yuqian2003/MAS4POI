import os
import json
import hashlib
import logging 
from tqdm import tqdm
from enum import Enum
from openai import OpenAI
from zhipuai import ZhipuAI
from tools.API import Spark, ERNIE, Llama
from transformers import pipeline, AutoTokenizer
from transformers.pipelines import SummarizationPipeline
"""

首先，当reflector agent发现这个结果是错误的时候（即在manage agent 的workflow函数中，
if correct： 判断失败，则使用reflector agent进行错误纠正）使用reflector agent进行错误纠正，然后，在这个错误纠正中，reflector agent会先调用LLM进行错误分析，然后把这个错误原因和错误结果一起存储到自身的记忆模块（self.memort = []）中，然后把这个memory当作llm的输入进行错误纠正，如果修改后仍然存在错误，则会继续纠正，
reflector agent会有一个attempt的变量，这个变量是用来记录纠错的次数的，如果纠错次数大于3次，则不进行纠错，还是按照原本item agent的输出，
然后在纠错后会保存自身的memory list，当成文本生成在终端中，然后调用清除缓存的函数对这个memory进行清空

目前达到：
分析错误并生成反思（reflection）
根据反思修改预测策略或候选集
保持反思和修改策略的历史记录

但是         ----------------------- 目前反思和策略修改的逻辑依赖于 LLM 的输出。为了提高鲁棒性，可以考虑在一定条件下使用预设的策略。
 ------------------------------------ 塞一个总结文本的模型
"""
### 应该设计一个开关，如果有，则启动refelctor agent，否则就不启动
### reflect 生成的文件中信息太少只有input，其实整体main生成的信息也太少

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReflectionStrategy(Enum):
    NONE = 'none'  
    REFLECTION = 'reflection'  
    LAST_ATTEMPT = 'last_attempt'  
    LAST_ATTEMPT_AND_REFLECTION = 'last_attempt_and_reflection'  

class ReflectorAgent:
    def __init__(self, llm, logger, reflection_strategy=ReflectionStrategy.REFLECTION, keep_reflections=True):
        self.setup_environment()
        self.reflection_strategy = reflection_strategy  
        self.keep_reflections = keep_reflections  
        self.memory = []
        self.attempt = 0
        self.logger = logger
        self.llm = self.set_llm(llm)
        # self.temperature = temperature
    
    def set_llm(self, llm_name):
        if llm_name == 'gpt':
            self.logger.info("Setting up GPT model...")
            return 'gpt' 
        elif llm_name == 'claude':
            self.logger.info("Setting up Claude model...")
            return 'claude'
        elif llm_name == 'spark':
            self.logger.info("Setting up Spark model...")
            return 'spark'
        elif llm_name == 'qwen':
            self.logger.info("Setting up Qwen model...")
            return 'qwen'
        elif llm_name == 'ernie':
            self.logger.info("Setting up Ernie model...")
            return 'ernie'
        elif llm_name == 'glm':
            self.logger.info("Setting up GLM model...")
            return 'glm'
        else:
            self.logger.error(f"Unknown LLM model: {llm_name}. Defaulting to GPT model.")
            return 'gpt'
    def setup_environment(self):
        os.environ["OPENAI_API_KEY"] = "sk-dfrFMkQ4O7U9t8tD7dD665854eC04c70900dD4Ea79732029"
        os.environ["OPENAI_API_BASE"] = "https://api.bltcy.ai/v1"

    def analyze_error(self, input, scratchpad):
        prompt = (
            f"Input: {json.dumps(input)}\n"
            f"Scratchpad: {json.dumps(scratchpad)}\n"
            f"Please analyze this process, identify any potential issues, and propose a plan for improvement."
        )
        messages=[{"role": "user", "content": prompt}]
            
        if self.llm == 'gpt':
            client = OpenAI(
                api_key = "sk-dfrFMkQ4O7U9t8tD7dD665854eC04c70900dD4Ea79732029",
                base_url = "https://api.bltcy.ai/v1"
            )
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0,
            )
            
        elif self.llm == 'llama':
            LLM = Llama
            response = LLM(messages)
            
        elif self.llm == 'spark':
            LLM = Spark
            response = LLM(messages)
            
        elif self.llm == 'qwen':
            client = OpenAI(
                api_key = "sk-07tA8QW5E8tBWceI5843BaAfDd8b48B0816112Ac19305081",
                base_url = "https://api.bltcy.ai/v1"
            )
            response = client.chat.completions.create(
                model='qwen-turbo',
                messages=messages,
                temperature=0,
            )
            
        elif self.llm == 'ernie':
            LLM = ERNIE
            response = LLM(messages)
            
        elif self.llm == 'glm':
            client = ZhipuAI(api_key="sk-yfYaDn1BEGUSwnuIDf3f1392Be7f4c9c90Be06Ea7e5cBb20")
            response = client.chat.completions.create(
                model="glm-3-turbo",  
                messages=messages,
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm}") 
        reflection = response.choices[0].message.content.strip()
        # logger.info(f"Reflection generated: {reflection}")
        self.memory.append({"input": input, "scratchpad": scratchpad, "reflection": reflection})
        return reflection
    
    def modify_based_on_reflection(self, reflection, scratchpad):
        prompt = (
            f"Given the reflection: {json.dumps(reflection)}\n"
            f"And the previous scratchpad: {json.dumps(scratchpad)}\n"
            "Please propose a modified prediction strategy or candidate selection to correct the error."
        )
        messages=[{"role": "user", "content": prompt}]
        if self.llm == 'gpt':
            client = OpenAI(
                api_key = "sk-dfrFMkQ4O7U9t8tD7dD665854eC04c70900dD4Ea79732029",
                base_url = "https://api.bltcy.ai/v1"
            )
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0,
            )
            
        elif self.llm == 'llama':
            LLM = Llama
            response = LLM(messages)
            
        elif self.llm == 'spark':
            LLM = Spark
            response = LLM(messages)
            
        elif self.llm == 'qwen':
            client = OpenAI(
                api_key = "sk-07tA8QW5E8tBWceI5843BaAfDd8b48B0816112Ac19305081",
                base_url = "https://api.bltcy.ai/v1"
            )
            response = client.chat.completions.create(
                model='qwen-turbo',
                messages=messages,
                temperature=0,
            )
            
        elif self.llm == 'ernie':
            LLM = ERNIE
            response = LLM(messages)
            
        elif self.llm == 'glm':
            client = ZhipuAI(api_key="sk-yfYaDn1BEGUSwnuIDf3f1392Be7f4c9c90Be06Ea7e5cBb20")
            response = client.chat.completions.create(
                model="glm-3-turbo",  
                messages=messages,
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm}")   
        modified_prediction = response.choices[0].message.content.strip()
        # logger.info(f"Modified prediction generated: {modified_prediction}")
        self.memory.append({"reflection": reflection, "modified_prediction": modified_prediction})
        return modified_prediction
    

    def forward(self, input, scratchpad, correct: bool):
        if correct:
            logger.info("Successfully correct!")
            return scratchpad  # Return original prediction

        if self.attempt >= 3:
            logger.info("Maximum attempts reached, returning original prediction.")
            return scratchpad

        self.attempt += 1

        reflection = self.analyze_error(input, scratchpad)
        modified_prediction = self.modify_based_on_reflection(reflection, scratchpad)

        return modified_prediction

    def clear_memory(self):
        self.memory = []
        self.attempt = 0

    def output_memory(self):
        reflect_dir = './reflect'
        os.makedirs(reflect_dir, exist_ok=True)

        for idx, item in enumerate(self.memory):
            file_path = os.path.join(reflect_dir, f"reflection_{idx + 1}.json")

            with open(file_path, 'w') as file:
                json.dump(item, file, indent=4)

            logger.info(f"Memory content saved to {file_path}")

    def process_errors(self, err_list):
        return sorted(err_list)

    def output_results(self, acc1, acc5, acc10, mrr, err):
        logger.info(f"ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}")
        with open('./testERR', 'a') as file:
            file.write(json.dumps(err, indent=4))