import os
import json
import hashlib
import logging 
from tqdm import tqdm
from enum import Enum
from openai import OpenAI
from zhipuai import ZhipuAI
from transformers import pipeline, AutoTokenizer
from transformers.pipelines import SummarizationPipeline
from typing import List, Dict, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

class ReflectorAgent:
    
    """
    ReflectorAgent class that uses different reflection strategies to refine POI recommendations
    using LLM (Language Learning Models) like GPT, Gemini, Moonshot, etc.

    Parameters:
    - llm (str): The LLM model name.
    - seed (int): Seed for reproducibility.
    - datasetName (str): Name of the dataset being used.
    - logger (object): Logger instance for logging information and errors.
    - times (int): Number of times to attempt reflection.
    - keep_reflections (bool): Whether to retain reflection history in memory.
    """
    def __init__(self, llm: str, seed: int, datasetName: str, logger, times: int, 
                 key: str, base: str,
                 keep_reflections: bool = True):
        self.keep_reflections = keep_reflections
        self.memory = []
        self.times = times
        self.attempt = 0
        self.logger = logger
        self.datasetName = datasetName
        self.seed = seed
        self.llm = self.set_llm(llm)

        # OPENAI
        self.key = key
        self.base = base
    
    def set_llm(self, llm_name: str) -> str:
        """
        Sets the LLM model based on the provided name and logs the model setup process.

        Parameters:
        - llm_name (str): Name of the LLM model.

        Returns:
        - str: The LLM model name.
        """
        llm_models = {
            'gpt': 'GPT model',
            'gemini': 'Gemini model',
            'moonshot': 'MoonShot model',
            'qwen': 'Qwen model',
            'claude': 'Claude model'
        }

        if llm_name in llm_models:
            self.logger.info(f"Reflector starting {llm_models[llm_name]}...")
            return llm_name
        else:
            self.logger.error(f"Unknown LLM model: {llm_name}. Defaulting to GPT model.")
            return 'gpt'
        

    def analyze_error(self, input_data: dict, scratchpad: dict) -> dict:
        """
        Analyzes the error in the recommendations and provides a reflection based on the input data.

        Parameters:
        - input_data (dict): The input data used for recommendation.
        - scratchpad (dict): The scratchpad containing the initial recommendation.

        Returns:
        - dict: Reflection containing analysis and improvements in JSON format.
        """
        prompt = (
            f"Input: {json.dumps(input)}\n"
            f"Scratchpad: {json.dumps(scratchpad)}\n\n"
            f"Task: Identify why the recommended POIs are incorrect and provide specific, actionable improvements in a JSON format. "
            f"Return a JSON object with keys such as 'issues' (listing detected problems) and 'improvements' (providing concrete steps to improve the recommendation).\n"
            f"Focus on how to use context like location, time, and user history to make better POI recommendations."
        )
        response_content = self.call_llm_api(prompt)

        try:
            reflection = json.loads(response_content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse reflection JSON: {e}")
            reflection = {}
        
        self.memory.append({"input": input_data, "scratchpad": scratchpad, "reflection": reflection})
        return reflection
    
    def modify_based_on_reflection(self, reflection: dict, scratchpad: dict) -> dict:
        """
        Modifies the POI recommendation based on the reflection provided.

        Parameters:
        - reflection (dict): The reflection containing issues and improvements.
        - scratchpad (dict): The initial recommendation.

        Returns:
        - dict: The modified recommendation.
        """
        prompt = (
            f"Based on the reflection analysis provided:\n"
            f"{json.dumps(reflection)}\n\n"
            f"Review the previous prediction details:\n"
            f"{json.dumps(scratchpad)}\n\n"
            "Using the insights from the reflection, generate a new set of POI recommendations that corrects the issues identified. "
            "Ensure that the new recommendations are more accurate and relevant by appropriately considering factors such as user location, time of day, and user history."
            "PAY MORE ATTENTION, You only need to generate the next POI recommendation, without providing any explanation."
        )
        response_content = self.call_llm_api(prompt)

        try:
            modified_prediction = json.loads(response_content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode modified prediction JSON: {e}")
            modified_prediction = scratchpad
        
        self.memory.append({"reflection": reflection, "modified_prediction": modified_prediction})
        return modified_prediction
    
    def call_llm_api(self, prompt: str) -> str:
        """
        Calls the LLM API with the provided prompt and returns the response content.

        Parameters:
        - prompt (str): The prompt to be sent to the LLM API.

        Returns:
        - str: The response content from the LLM API.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            client = OpenAI(
                api_key=os.getenv(self.key),
                base_url=os.getenv(self.base)
            )
            response = client.chat.completions.create(
                model=self.get_llm_model(),
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            return ""
    
    def get_llm_model(self) -> str:
        """
        Retrieves the correct LLM model based on the selected LLM.

        Returns:
        - str: The LLM model identifier.
        """
        llm_model_mapping = {
            'gpt': 'gpt-3.5-turbo',
            'gemini': 'gemini-pro',
            'moonshot': 'moonshot-v1-8k',
            'qwen': 'qwen-turbo',
            'claude': 'claude-3-5-sonnet-20240620',
            'llama': 'llama-2-70b'
        }
        return llm_model_mapping.get(self.llm, 'gpt-3.5-turbo')
    

    def forward(self, input_data: dict, scratchpad: dict, correct: bool) -> dict:
        """
        Executes the reflection process and returns the modified prediction.

        Parameters:
        - input_data (dict): The input data used for recommendation.
        - scratchpad (dict): The initial recommendation.
        - correct (bool): Flag indicating if the initial prediction was correct.

        Returns:
        - dict: The final recommendation after reflection and modification.
        """
        if correct:
            self.logger.info("Successfully correct!")
            return scratchpad

        reflection = self.analyze_error(input_data, scratchpad)
        self.logger.info(f'Reflection Content: {reflection}')
        modified_prediction = self.modify_based_on_reflection(reflection, scratchpad)
        self.logger.info(f'This is {self.attempt} REFINE: {modified_prediction}')
        self.attempt += 1

        return modified_prediction

    def clear_memory(self) -> None:
        """
        Clears the reflection memory and resets the attempt counter.
        """
        self.memory = []
        self.attempt = 0

    def output_memory(self) -> None:
        """
        Outputs the memory of reflections and predictions to JSON files for later analysis.
        """
        reflect_dir = './reflect'
        os.makedirs(reflect_dir, exist_ok=True)

        llm_dir = os.path.join(reflect_dir, self.datasetName, self.llm, str(self.seed))
        os.makedirs(llm_dir, exist_ok=True)

        for idx, item in enumerate(self.memory):
            file_path = os.path.join(llm_dir, f"reflector_{idx + 1}.json")
            with open(file_path, 'w') as file:
                json.dump(item, file, indent=4)

    def process_errors(self, err_list: List[Any]) -> List[Any]:
        """
        Processes and sorts a list of errors for analysis.

        Parameters:
        - err_list (List[Any]): The list of errors to process.

        Returns:
        - List[Any]: The sorted list of errors.
        """
        return sorted(err_list)

    def output_results(self, acc1: float, acc5: float, acc10: float, mrr: float, err: List[Any]) -> None:
        """
        Outputs the results of the prediction process including accuracy and errors.

        Parameters:
        - acc1 (float): Accuracy at rank 1.
        - acc5 (float): Accuracy at rank 5.
        - acc10 (float): Accuracy at rank 10.
        - mrr (float): Mean reciprocal rank.
        - err (List[Any]): List of errors encountered.
        """
        self.logger.info(f"ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}")
        with open('./testERR', 'a') as file:
            file.write(json.dumps(err, indent=4))

    def call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Calls the LLM API with the provided messages.

        Parameters:
        - messages (List[Dict[str, str]]): List of messages to send to the LLM.

        Returns:
        - Dict[str, Any]: The LLM's response parsed as a JSON object.
        """
        try:
            if self.llm == 'gpt':
                client = OpenAI(api_key=os.getenv(self.key), base_url=os.getenv(self.base))
                response = client.chat.completions.create(model='gpt-3.5-turbo', messages=messages, temperature=0)
            # Repeat similar blocks for other LLMs (gemini, moonshot, qwen, claude, llama)
            else:
                raise ValueError(f"Unsupported LLM model: {self.llm}")

            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            return {}
        
    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Saves data to a JSON file.

        Parameters:
        - data (Dict[str, Any]): The data to save.
        - file_path (str): The path to the file where data will be saved.
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            self.logger.info(f"Data saved successfully to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {e}")

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Loads data from a JSON file.

        Parameters:
        - file_path (str): The path to the file from which data will be loaded.

        Returns:
        - Dict[str, Any]: The loaded data.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            self.logger.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            return {}
        
    def validate_json_format(self, data: str) -> bool:
        """
        Validates whether a string is a properly formatted JSON.

        Parameters:
        - data (str): The string to validate.

        Returns:
        - bool: True if the string is valid JSON, False otherwise.
        """
        try:
            json.loads(data)
            return True
        except json.JSONDecodeError:
            return False
        
    def check_directory_exists(self, directory_path: str) -> bool:
        """
        Checks if a directory exists and creates it if it doesn't.

        Parameters:
        - directory_path (str): The path of the directory to check.

        Returns:
        - bool: True if the directory exists or was created successfully, False otherwise.
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory_path}: {e}")
            return False