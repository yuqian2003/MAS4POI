import random
import logging
from tqdm import tqdm
from agents.Navigator import NaviAgent
from typing import List, Dict, Union, Any, Tuple
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
            data = self.data_agent.process()
            self.logger.info("Data loaded successfully through Data Agent.")
            
            hit1, hit5, hit10, rr = 0, 0, 0, 0
            err = []
            reflector_agent = ReflectorAgent(llm=self.llm, logger=self.logger, times=self.times, 
                                             datasetName=self.datasetName, seed=self.seed, 
                                             key=self.key, base=self.base)
            self.logger.info("Reflector agent initialized.")

            for trajectory, groundTruth in tqdm(data["targets"].items()):
                self.logger.info(f'Processing trajectory: {trajectory}')
                candidateSet = self.generate_candidate_set(trajectory, groundTruth, data)

                try:
                    prediction = self.analyst.generate_recommendation(
                        trajectory, candidateSet, groundTruth,
                        data["longs"], data["recents"], data["traj2u"], data["poiInfos"]
                    )
                    correct = self.process_prediction(trajectory, groundTruth, prediction, reflector_agent, candidateSet, data)
                    
                    if correct:
                        hit1, hit5, hit10, rr = self.update_metrics(hit1, hit5, hit10, rr, prediction, groundTruth)
                    else:
                        err.append((trajectory, groundTruth, candidateSet, data["longs"], data["recents"], data["traj2u"], data["poiInfos"]))
                        self.logger.error(f'Failed to correct prediction for trajectory {trajectory} after {self.times} attempts.')

                except Exception as e:
                    self.logger.error(f'Error encountered for trajectory {trajectory}: {repr(e)}')
                self.logger.info('-' * 100)

            return self.finalize_poi_workflow(reflector_agent, hit1, hit5, hit10, rr, len(data["targets"]), err)

        except Exception as e:
            self.logger.error(f"Error in POI workflow: {e}")
            return 0.0, 0.0, 0.0, 0.0

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
            negSample = random.sample(data["poiList"], self.case_num)
            candidateSet = negSample + [groundTruth[0]]
            return candidateSet
        except Exception as e:
            self.logger.error(f"Error generating candidate set: {e}")
            return []

    def process_prediction(self, trajectory: str, groundTruth: tuple, prediction: list, 
                           reflector_agent, candidateSet: list, data: dict) -> bool:
        """
        Processes the prediction and applies reflection if needed.

        Parameters:
        - trajectory (str): The trajectory ID.
        - groundTruth (tuple): The ground truth POI for the trajectory.
        - prediction (list): The predicted POIs.
        - reflector_agent: The agent responsible for reflection.
        - candidateSet (list): The set of candidate POIs.
        - data (dict): The dataset containing POIs and trajectory information.

        Returns:
        - bool: Whether the prediction was correct or not.
        """
        try:
            correct = groundTruth[0] in prediction
            if correct:
                index = prediction.index(groundTruth[0]) + 1
                self.logger.info(f'Correct prediction at position {index}')
            else:
                self.logger.warning(f'Initial prediction failed for trajectory {trajectory}')
                correct = self.apply_reflection(trajectory, reflector_agent, candidateSet, groundTruth, correct)
            return correct
        except Exception as e:
            self.logger.error(f"Error processing prediction: {e}")
            return False

    def apply_reflection(self, trajectory: str, reflector_agent, candidateSet: list, 
                         groundTruth: tuple, correct: bool) -> bool:
        """
        Applies reflection to correct a failed prediction.

        Parameters:
        - trajectory (str): The trajectory ID.
        - reflector_agent: The agent responsible for reflection.
        - candidateSet (list): The set of candidate POIs.
        - groundTruth (tuple): The ground truth POI for the trajectory.
        - correct (bool): Whether the initial prediction was correct or not.

        Returns:
        - bool: Whether the prediction was corrected after reflection or not.
        """
        try:
            for i in range(self.times):
                self.logger.info(f'Attempt {i + 1}: Reflector agent processing...')
                prediction = reflector_agent.forward(trajectory, candidateSet, correct)
                correct = groundTruth[0] in prediction
                if correct:
                    self.logger.info('Corrected by reflector agent.')
                    break
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
            reflector_agent.output_memory()
            reflector_agent.clear_memory()

            acc1 = hit1 / num_trajectories
            acc5 = hit5 / num_trajectories
            acc10 = hit10 / num_trajectories
            mrr = rr / num_trajectories

            self.logger.info(f"Workflow completed. ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}")
            return acc1, acc5, acc10, mrr
        except Exception as e:
            self.logger.error(f"Error finalizing POI workflow: {e}")
            return 0.0, 0.0, 0.0, 0.0