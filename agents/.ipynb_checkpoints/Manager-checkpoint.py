import random
import logging
from agents.Reflector import ReflectorAgent
from tqdm import tqdm

class ManageAgent:
    def __init__(self, data_agent, item_agent, search_agent, navigator_agent, task, seed_value, case_num, logger, llm):
        self.data_agent = data_agent
        self.item_agent = item_agent
        self.search_agent = search_agent
        self.navigator_agent = navigator_agent
        self.task = task
        self.seed_value = seed_value
        self.case_num = case_num
        self.logger = logger
        self.llm = self.set_llm(llm)

        logger.propagate = False

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

    def run_workflow_without_reflector(self):
        if self.task == 'poi':
            print("Starting POI recommendation workflow...", flush=True)
            data = self.data_agent.process()
            print("Data loaded successfully through Data Agent.", flush=True)
            hit1 = hit5 = hit10 = rr = 0
            err = []
            for trajectory, groundTruth in tqdm(data["targets"].items()):
                trajectory = str(trajectory)
                seed_value = eval(trajectory)
                random.seed(seed_value)
                negSample = random.sample(data["poiList"], self.case_num)   
                candidateSet = negSample + [groundTruth[0]]
                try:
                    prediction = self.item_agent.generate_recommendation(
                        trajectory, candidateSet, groundTruth,
                        data["longs"], data["recents"], data["traj2u"], data["poiInfos"]
                    )
                    if groundTruth[0] in prediction:
                        index = prediction.index(groundTruth[0]) + 1
                        if index == 1:
                            hit1 += 1
                        if index <= 5:
                            hit5 += 1
                        hit10 += 1
                        rr += 1 / index
                    else:
                        err.append(eval(trajectory))
                except Exception as e:
                    tqdm.write(repr(e))

            num_trajectories = len(data["targets"])
            tqdm.write(f"This is the number of trajectories: {num_trajectories}")
            tqdm.write("This is the number of hit1-{}, hit5-{}, hit10-{}".format(hit1, hit5, hit10))
            acc1 = hit1 / num_trajectories
            acc5 = hit5 / num_trajectories
            acc10 = hit10 / num_trajectories
            mrr = rr / num_trajectories

            tqdm.write(f"Workflow completed. ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}")
            return acc1, acc5, acc10, mrr
        elif self.task == 'navigator':
            print("Start Navigator Agent")
            pass
        elif self.task == 'search':
            print("Start Search Agent")
            pass
        else:
            pass

    def run_workflow(self):
        if self.task == 'poi':
            print("Starting POI recommendation workflow...", flush=True)
            data = self.data_agent.process()
            print("Data loaded successfully through Data Agent.", flush=True)
            hit1 = hit5 = hit10 = rr = 0
            err = []

            reflector_agent = ReflectorAgent(llm=self.llm,logger = self.logger)
            self.logger.info("Reflector agent initialized.")

            for trajectory, groundTruth in tqdm(data["targets"].items()):

                self.logger.info(f'Processing trajectory: {trajectory}')
                self.logger.info(f'Ground truth: {groundTruth}')

                trajectory = str(trajectory)
                seed_value = eval(trajectory)
                random.seed(seed_value)
                negSample = random.sample(data["poiList"], self.case_num)
                candidateSet = negSample + [groundTruth[0]]
                # self.logger.info(f'Candidate set: {candidateSet}')

                try:
                    prediction = self.item_agent.generate_recommendation(
                        trajectory, candidateSet, groundTruth,
                        data["longs"], data["recents"], data["traj2u"], data["poiInfos"]
                    )

                    self.logger.info(f'Prediction: {prediction}')

                    correct = groundTruth[0] in prediction
                    if correct:
                        index = prediction.index(groundTruth[0]) + 1
                        self.logger.info(f'Correct prediction at position {index}')
                        if index == 1:
                            hit1 += 1
                        if index <= 5:
                            hit5 += 1
                        hit10 += 1
                        rr += 1 / index
                    else:
                        self.logger.warning(f'Initial prediction failed for trajectory {trajectory}')
                        for i in range(3):
                            self.logger.info(f'Attempt {i + 1}: Reflector agent processing...')
                            prediction = reflector_agent.forward(trajectory, candidateSet, correct)
                            # self.logger.info(f'Prediction after reflection: {prediction}')
                            correct = groundTruth[0] in prediction
                            if correct:
                                self.logger.info('Corrected by reflector agent.')
                                break
                        if not correct:
                            err.append((trajectory, groundTruth, candidateSet, data["longs"], data["recents"], data["traj2u"], data["poiInfos"]))
                            self.logger.error(f'Failed to correct prediction for trajectory {trajectory} after 3 attempts.')
                except Exception as e:
                    self.logger.error(f'Error encountered for trajectory {trajectory}: {repr(e)}')
                self.logger.info('-' * 50)  # 分割线

            reflector_agent.output_memory()

            reflector_agent.clear_memory()

            num_trajectories = len(data["targets"])
            acc1 = hit1 / num_trajectories
            acc5 = hit5 / num_trajectories
            acc10 = hit10 / num_trajectories
            mrr = rr / num_trajectories

            self.logger.info(f"Workflow completed. ACC@1: {acc1:.4f}, ACC@5: {acc5:.4f}, ACC@10: {acc10:.4f}, MRR: {mrr:.4f}")
            return acc1, acc5, acc10, mrr

        elif self.task == 'navigator':
            self.logger.info("Start Navigator Agent")
            pass
        elif self.task == 'search':
            self.logger.info("Start Search Agent")
            pass
        else:
            pass