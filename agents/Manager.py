import random
import json
import logging
from tqdm import tqdm
from collections import defaultdict
from agents.Navigator import NaviAgent
from typing import List, Dict, Union, Any, Tuple, Optional
from agents.Reflector import ReflectorAgent

class ManageAgent:
    def __init__(self, data_agent, analyst, search_agent, navi_agent, 
                 task: str, seed: int, case_num: int, logger, llm: str, times: int, datasetName: str, 
                 amap_api_key: str, source: Union[str, Tuple[float, float]], 
                 target: Union[str, Tuple[float, float]], city: str, use_coordinates: bool, key: str, base: str):
        """
        Initializes the ManageAgent class, setting up the necessary agents, configurations, and logging.

        Parameters:
        - data_agent: The agent responsible for data handling.
        - analyst: The agent responsible for making predictions.
        - search_agent: The agent responsible for search-related tasks.
        - navi_agent: The agent responsible for navigation tasks.
        - task (str): The task type, can be 'poi', 'navigator', or 'search'.
        - seed (int): Random seed for reproducibility.
        - case_num (int): Number of cases for testing.
        - logger: Logger for tracking workflow.
        - llm (str): The chosen LLM model.
        - times (int): Number of reflection attempts.
        - datasetName (str): Name of the dataset being used.
        - amap_api_key (str): API key for the navigation service.
        - source (Union[str, Tuple[float, float]]): The source location.
        - target (Union[str, Tuple[float, float]]): The target location.
        - city (str): The city name.
        - use_coordinates (bool): Whether to use coordinates or addresses.
        - key (str): API key for external services.
        - base (str): Base URL for external services.
        """
        # various agents
        self.data_agent = data_agent
        self.analyst = analyst
        self.search_agent = search_agent
        self.navi_agent = navi_agent
        self.llm = None

        # Reflector
        self.times = times
        
        # Data Agent
        self.datasetName = datasetName
        

        # Navigator
        self.amap_api_key = amap_api_key 
        self.source = source
        self.target = target
        self.city = city
        self.use_coordinates = use_coordinates

        # logger
        self.task = task
        self.seed = seed
        self.case_num = case_num
        self.logger = logger
        logger.propagate = False
        
        # Initial
        self.llm = self.set_llm(llm)
        self.key = key
        self.base = base
    
    def set_llm(self, llm_name: str) -> str:
        """
        Sets the LLM model based on the provided name and logs the model setup process.

        Parameters:
        - llm_name (str): The name of the LLM model.

        Returns:
        - str: The name of the LLM model that was set.
        """
        try:
            llm_models = {
                'gpt': 'GPT model',
                'gemini': 'Gemini model',
                'moonshot': 'MoonShot model',
                'qwen': 'Qwen model',
                'claude': 'Claude model'
            }

            if llm_name in llm_models:
                self.logger.info(f"Manager starting {llm_models[llm_name]}...")
                return llm_name
            else:
                self.logger.error(f"Unknown LLM model: {llm_name}. Defaulting to GPT model.")
                return 'gpt'
        except Exception as e:
            self.logger.error(f"Error setting LLM model: {e}")
            return 'gpt'
    def run_workflow(self) -> Union[Tuple[float, float, float, float], None]:
        """
        Runs the workflow for the specified task (poi, navigator, or search).

        Returns:
        - Tuple[float, float, float, float]: Metrics (ACC@1, ACC@5, ACC@10, MRR) for the POI workflow.
        - None: For tasks other than 'poi'.
        """
        try:
            if self.task == 'poi':
                return self.run_poi_workflow()
            elif self.task == 'navigator':
                self.run_navigator_workflow()
            elif self.task == 'search':
                self.run_search_workflow()
            else:
                self.logger.error(f"Unknown task: {self.task}")
        except Exception as e:
            self.logger.error(f"Error running workflow: {e}")
            return None

    def run_poi_workflow(self) -> Tuple[float, float, float, float]:
        """
        Executes the workflow for POI recommendation.

        Returns:
        - Tuple[float, float, float, float]: Metrics (ACC@1, ACC@5, ACC@10, MRR).
        """
        try:
            self.logger.info("Starting POI recommendation workflow...")
            data = self.data_agent.process()   # 在此处通过data agent获取数据
            """
            return {
                "longs": longs,
                "recents": recents,
                "targets": targets,
                "poiInfos": poiInfos,
                "traj2u": traj2u,
                "poiList": list(poiInfos.keys())
            }
            """
            self.logger.info("Data loaded successfully through Data Agent.")
            
            hit1, hit5, hit10, rr = 0, 0, 0, 0
            err = []
            err_cat = []
            reflector_agent = ReflectorAgent(llm=self.llm, logger=self.logger, times=self.times, 
                                             datasetName=self.datasetName, seed=self.seed, 
                                             key=self.key, base=self.base, temperature = 0.)
            self.logger.info("Reflector agent initialized.")

            for trajectory, groundTruth in tqdm(data["targets"].items()):
                self.logger.info(f'Processing trajectory: {trajectory}')
                candidateSet = self.generate_candidate_set(trajectory, groundTruth, data)  # candidate set, data comes from Data agent
                attempt = 0
                try:
                    prediction, longterm, recent, candidates = self.analyst.generate_recommendation(
                        trajectory, candidateSet, groundTruth,
                        data["longs"], data["recents"], data["traj2u"], data["poiInfos"]
                    )

                    u = data["traj2u"].get(trajectory) # traj2u --> user id
                    long = data["longs"].get(u, [])  # longs
                    rec = data["recents"].get(trajectory, []) # recents
                    time = rec[-1][1]
                    
                    # reflector input
                    input_data = {"longterm": longterm, "recent": recent, "candidates": candidates, 
                                  "trajectory": trajectory, "time": time, "u": u}
                    # process_prediction input --> 
                    """
                    Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based 
                    on his/her trajectory information.
                    <question> The following is a trajectory of user {user_id}: {recent}. \
                    There is also historical data: {longterm}. Given the data, at {time}, which POI id \
                    will user {user_id} visit? Note that POI id is an identifier in the set of POIs. \
                    <answer>: At {time}, user {u} will visit POI id {poi_id}.
                    """
                    correct = self.process_prediction(trajectory, groundTruth, prediction, reflector_agent, input_data, attempt)
                    
                    
                    pre_cat = data["poiInfos"].get(prediction[0], {}).get("category", "Unknown")
                    true_cat = data["poiInfos"].get(groundTruth[0], {}).get("category", "Unknown")
                    
                    
                    if correct:
                        hit1, hit5, hit10, rr = self.update_metrics(hit1, hit5, hit10, rr, prediction, groundTruth)
                    else:
                        err.append((trajectory, groundTruth, candidateSet, data["longs"], data["recents"], data["traj2u"], data["poiInfos"]))
                        err_cat.append({"pre":pre_cat,
                                       "true_Cat":true_cat,
                                       "info":data["poiInfos"]})
                        self.logger.error(f'Failed to correct prediction for trajectory {trajectory} after {attempt + 1} attempts.')
                            
                except Exception as e:
                    self.logger.error(f'Error encountered for trajectory {trajectory}: {repr(e)}')
                self.logger.info('-' * 100)
            
            self.save_errors_to_file(err_cat, output_path='poi_errors.json') # categorical statistic
            return self.finalize_poi_workflow(reflector_agent, hit1, hit5, hit10, rr, len(data["targets"]), err)

        except Exception as e:
            self.logger.error(f"Error in POI workflow: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def save_errors_to_file(self, err: List[Dict], output_path: str = 'poi_errors.json') -> None:
        """
        Saves the list of errors to a JSON file for further analysis.
        
        Args:
        - err: A list of error details, each being a dictionary containing information about the error.
        - output_path: The path where the JSON file will be saved.
        """
        print(err)
        self.logger.info(f"Attempting to save errors to {output_path}")
        try:
            with open(output_path, 'w') as file:
                json.dump(err, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Errors saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving errors to file: {e}")

    def run_navigator_workflow(self) -> None:
        """
        Executes the workflow for the navigator task.
        """
        try:
            self.logger.info("Starting NAVIGATOR task.")
            self.navi_agent.answer()
        except Exception as e:
            self.logger.error(f"Error in Navigator workflow: {e}")

    def run_search_workflow(self) -> None:
        """
        Executes the search workflow using the SearcherAgent.
        """
        try:
            self.logger.info("Starting SEARCH task.")
            # Instantiate the SearcherAgent
            # Call the SearcherAgent's method to perform the search
            response = self.search_agent.get_response()
            self.logger.info(f"Search result: {response}")
        
        except Exception as e:
            self.logger.error(f"Error in Search workflow: {e}")

    def generate_candidate_set(self, trajectory: str, groundTruth: tuple, data: dict) -> List[str]:
        """
        Generates a candidate set of POIs for the given trajectory.

        Parameters:
        - trajectory (str): The trajectory ID.
        - groundTruth (tuple): The ground truth POI for the trajectory.
        - data (dict): The dataset containing POIs.

        Returns:
        - List[str]: A list of candidate POIs.
        """
        try:
            seed_value = eval(trajectory)
            random.seed(seed_value)
            negSample = random.sample(data["poiList"], 100)  ############  在这里生成随机的100个sample, data 源于data agent
            candidateSet = negSample + [groundTruth[0]]
            return candidateSet
        except Exception as e:
            self.logger.error(f"Error generating candidate set: {e}")
            return []

    def process_prediction(self, trajectory: str, groundTruth: tuple, prediction: list, 
                           reflector_agent, input_data: dict, attempt: int) -> bool:
        """
        Processes the prediction and applies reflection if needed.  the prediction comes from Analyst

        Parameters:
        - trajectory (str): The trajectory ID.
        - groundTruth (tuple): The ground truth POI for the trajectory.
        - prediction (list): The predicted POIs.  comes from analyst
        - reflector_agent: The agent responsible for reflection.
        - data (dict): The dataset containing POIs and trajectory information.

        Returns:
        - bool: Whether the prediction was correct or not.
        """
        try:
            correct = groundTruth[0] in prediction
            if correct:
                index = prediction.index(groundTruth[0]) + 1  # idx
                self.logger.info(f'Correct prediction at position {index}')
            else:
                # reflector
                self.logger.warning(f'Initial prediction failed for trajectory {trajectory}')
                while not correct and attempt < self.times:
                    #  scratchpad --> analyst wrong answer
                    self.logger.info(f'Attempt {attempt + 1}: Reflector agent processing...')
                    correct = self.apply_reflection(groundTruth, reflector_agent, input_data, prediction, correct, attempt)
                    attempt += 1  # Increment attempt count after each reflection
            return correct
        except Exception as e:
            self.logger.error(f"Error processing prediction: {e}")
            return False

    def apply_reflection(self, groundTruth, reflector_agent, input_data, 
                         scratchpad, correct, attempt) -> bool:
        """
        Applies reflection to correct a failed prediction.

        Parameters:
        - trajectory (str): The trajectory ID.
        - reflector_agent: The agent responsible for reflection.
        - candidateSet (list): The set of candidate POIs.
        - groundTruth (tuple): The ground truth POI for the trajectory.
        - correct (bool): Whether the initial prediction was correct or not.
        - attempt int --> record the reflect iteration
        Returns:
        - bool: Whether the prediction was corrected after reflection or not.
        """
        try:
            # for i in range(self.times):       not Loop
            
            # input data --> 
            # self.logger.info(f"Input Data: {input_data}")
            # self.logger.info(f"Scratchpad: {scratchpad}")
            # self.logger.info(f"Correct Flag: {correct}")
            
            
            # for key, value in input_data.items():
            #     if isinstance(value, str) and value.startswith("0") and len(value) > 1:
            #         raise ValueError(f"Invalid input data: {key} cannot start with a leading zero.")
            # list' object has no attribute 'items' clean_scratchpad
            
            
            # scratchpad = self.clean_scratchpad(scratchpad)
            # self.logger.info(f"****************************************************{input_data}")
            self.logger.info(f"*********************************************************************aaaaaaaaaaaaaaaaaaaaaa")
            # self.logger.info(scratchpad)
            ###### call the reflector agent
            
            # prediction = reflector_agent.forward(input_data, scratchpad, correct) 
            # self.logger.info(f"********************************************************************bbbbbbbbbbbbbbbbbbbbbbb")
            ##  try to divide
            reflection = reflector_agent.analyze_error(input_data, scratchpad)
            self.logger.info(f'Reflection Content: {reflection}')
            self.logger.info(f"********************************************************************bbbbbbbbbbbbbbbbbbbbbbb")
            modified_prediction = reflector_agent.modify_based_on_reflection(input_data, reflection, scratchpad)
            self.logger.info(f'This is {attempt} REFINE: {modified_prediction}')
            self.logger.info(f"********************************************************************ccccccccccccccccccccccc")
            # self.attempt += 1 
            # only affect the reflector agent attribute
            # self.logger.info(f"************* {}prediction")
            # output by forward function in reflector agent --- > modified_prediction
            # self.logger.info(prediction)
            if not modified_prediction:
                self.logger.error("Received invalid modified_prediction from reflection process.")
                return False
                
            correct = groundTruth[0] in modified_prediction
            if correct:
                self.logger.info('Corrected by reflector agent.')
            else:
                self.logger.info(f'Attempt {attempt + 1} failed, trying again...')
            return correct
        except Exception as e:
            self.logger.error(f"Error applying reflection: {e}")
            return False

    def update_metrics(self, hit1: int, hit5: int, hit10: int, rr: float, 
                       prediction: list, groundTruth: tuple) -> Tuple[int, int, int, float]:
        """
        Updates the accuracy metrics based on the correct prediction.

        Parameters:
        - hit1 (int): The count of correct predictions at rank 1.
        - hit5 (int): The count of correct predictions within the top 5 ranks.
        - hit10 (int): The count of correct predictions within the top 10 ranks.
        - rr (float): The reciprocal rank.
        - prediction (list): The predicted POIs.
        - groundTruth (tuple): The ground truth POI for the trajectory.

        Returns:
        - Tuple[int, int, int, float]: Updated metrics (hit1, hit5, hit10, rr).
        """
        try:
            index = prediction.index(groundTruth[0]) + 1
            if index == 1:
                hit1 += 1
            if index <= 5:
                hit5 += 1
            hit10 += 1
            rr += 1 / index
            return hit1, hit5, hit10, rr
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
            return hit1, hit5, hit10, rr
    def clean_scratchpad(self, scratchpad):
        cleaned_list = []
        for entry in scratchpad:
            if isinstance(entry, str) and entry.startswith("0") and len(entry) > 1:
                cleaned_list.append(entry.lstrip("0"))  
            else:
                cleaned_list.append(entry)  
        return cleaned_list
    def finalize_poi_workflow(self, reflector_agent, hit1: int, hit5: int, hit10: int, 
                              rr: float, num_trajectories: int, err: list) -> Tuple[float, float, float, float]:
        """
        Finalizes the POI workflow by calculating metrics and saving reflections.

        Parameters:
        - reflector_agent: The agent responsible for reflection.
        - hit1 (int): The count of correct predictions at rank 1.
        - hit5 (int): The count of correct predictions within the top 5 ranks.
        - hit10 (int): The count of correct predictions within the top 10 ranks.
        - rr (float): The reciprocal rank.
        - num_trajectories (int): The number of processed trajectories.
        - err (list): List of failed predictions.

        Returns:
        - Tuple[float, float, float, float]: Final accuracy metrics (ACC@1, ACC@5, ACC@10, MRR).
        """
        try:
            # Obiect of type set is not JSON serializable
            self.logger.info(f"The hit1 is:{hit1}")
            self.logger.info(f"The hit5 is:{hit5}")
            self.logger.info(f"The hit10 is:{hit10}")
            self.logger.info(f"The rr is:{rr}")
            self.logger.info(f"The num_trajectories is:{num_trajectories}")
            reflector_memory = reflector_agent.output_memory()
            self.logger.info(f"The reflector_memory is:{reflector_agent.output_memory()}")
            acc1 = hit1 / num_trajectories
            acc5 = hit5 / num_trajectories
            acc10 = hit10 / num_trajectories
            mrr = rr / num_trajectories

            self.logger.info(f"Workflow completed. ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}")
            # if isinstance(reflector_memory, set):
            #     reflector_memory = list(reflector_memory)
            # reflector_agent.output_memory()
            # reflector_agent.clear_memory()

            return acc1, acc5, acc10, mrr
        except Exception as e:
            self.logger.error(f"Error finalizing POI workflow: {e}")
            return 0.0, 0.0, 0.0, 0.0
