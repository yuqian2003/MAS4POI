import os
import json
import logging
from openai import OpenAI
from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Tuple, Optional, Any

class Analyst:
    """
    ItemAgent is responsible for generating personalized recommendations for users
    based on their past trajectory data and candidate points of interest (POIs).

    The class supports multiple large language models (LLMs) and uses these models 
    to predict the next POI a user is likely to visit based on trajectory history.
    """

        def __init__(self, llm: str, logger: logging.Logger, temperature: float, prompt_id: str, key: str, base: str) -> None:
        self.logger = logger
        self.llm = self.set_llm(llm)
        self.temperature = temperature
        
        # OPENAI
        self.key = key
        self.base = base
        
        # prompt comparation
        self.prompt_id = prompt_id
    
    def set_llm(self, llm_name: str) -> str:
        """
        Sets the LLM model based on the provided name and logs the model setup process.
        """
        llm_models = {
            'gpt': 'GPT model',
            'gemini': 'Gemini model',
            'moonshot': 'MoonShot model',
            'qwen': 'Qwen model',
            'claude': 'Claude model'
        }

        if llm_name in llm_models:
            self.logger.info(f"Analyst starting {llm_models[llm_name]}...")
            return llm_name
        else:
            self.logger.error(f"Unknown LLM model: {llm_name}. Defaulting to GPT model.")
            return 'gpt'

    def haversine_distance(self, lat1: str, lon1: str, lat2: str, lon2: str) -> Optional[float]:
        """
        Calculates the Haversine distance between two geographic points.
        """
        try:
            lat1 = eval(lat1)
            lon1 = eval(lon1)
            lat2 = eval(lat2)
            lon2 = eval(lon2)
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            radius = 6371.0  # Earth radius in kilometers
            distance = radius * c
            return distance
        except Exception as e:
            self.logger.error(f"Error calculating Haversine distance: {e}")
            return None

    def generate_recommendation(
        self,
        trajectory: str,
        candidateSet: List[str],
        groundTruth: List[str],
        longs: Dict[str, List[Tuple[str, str]]],
        recents: Dict[str, List[Tuple[str, str]]],
        traj2u: Dict[str, str],
        poiInfos: Dict[str, Dict[str, str]]
    ) -> List[str]:
        """
        Generates a recommendation for the next point of interest (POI) a user might visit based on trajectory history.
        """
        try:
            u = traj2u.get(trajectory)
            if u is None:
                self.logger.error(f"User ID for trajectory {trajectory} not found.")
                return []
            
            long = longs.get(u, [])
            rec = recents.get(trajectory, [])

            if not self.validate_trajectory_data(long, rec, candidateSet, poiInfos):
                return []

            mostrec = rec[-1][0]
            longterm = [(poi, poiInfos[poi]["category"]) for poi, _ in long][-40:]
            recent = [(poi, poiInfos[poi]["category"]) for poi, _ in rec][-5:]
            
            candidates = self.create_candidates(mostrec, candidateSet, poiInfos)
            poi_id = candidates[0][0]  
            user_id = u  
            id_range = len(poiInfos)
            time = rec[-1][1] 
            if self.prompt_id == 'a':
                prompt = self.create_prompt_a(user_id, trajectory, candidateSet, longterm, recent, time)
            elif self.prompt_id == 'b':
                prompt = self.create_prompt_b(user_id, trajectory, candidateSet, longterm, recent, time)
            elif self.prompt_id == 'c':
                prompt = self.create_prompt_c(user_id, trajectory, poi_id, id_range, time, recent, longterm)
            elif self.prompt_id == 'd':
                prompt = self.create_prompt_d(longterm, recent, candidates)
            elif self.prompt_id == 'e':
                prompt = self.create_prompt_e(user_id, trajectory, candidateSet, longterm, recent, time)
            else:
                self.logger.error(f"Unknown prompt_id: {self.prompt_id}")
                return [] 
            # self.logger.info(f"Generated prompt for trajectory {trajectory}: {prompt}")
            prediction = self.call_llm_api(prompt)
            if not prediction:
                self.logger.warning(f"Prediction is empty for trajectory {trajectory}")
            # prediction = self.call_llm_api(prompt)
            output = self.create_output_data(prompt, prediction, groundTruth)
            self.save_response(output, trajectory)
            self.logger.info(f'This is the prediction by ANALYST: {prediction}')

            return prediction
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return []

    def validate_trajectory_data(
        self, 
        long: List[Tuple[str, str]], 
        rec: List[Tuple[str, str]], 
        candidateSet: List[str], 
        poiInfos: Dict[str, Dict[str, str]]
    ) -> bool:
        """
        Validates the trajectory data to ensure the necessary components are present.
        """
        if not long:
            self.logger.error("Long-term trajectory data is missing or empty.")
            return False
        if not rec:
            self.logger.error("Recent trajectory data is missing or empty.")
            return False
        if not candidateSet:
            self.logger.error("Candidate set is missing or empty.")
            return False
        if not poiInfos:
            self.logger.error("POI information is missing or empty.")
            return False
        return True

    def create_candidates(
        self, 
        mostrec: str, 
        candidateSet: List[str], 
        poiInfos: Dict[str, Dict[str, str]]
    ) -> List[Tuple[str, Optional[float], str]]:
        """
        Creates a list of candidate POIs with distances and categories.
        """
        try:
            candidates = [
                (poi, self.haversine_distance(poiInfos[poi]["latitude"], poiInfos[poi]["longitude"],
                                              poiInfos[mostrec]["latitude"], poiInfos[mostrec]["longitude"]),
                 poiInfos[poi]["category"])
                for poi in candidateSet
            ]
            candidates.sort(key=lambda x: x[1])
            return candidates
        except Exception as e:
            self.logger.error(f"Error creating candidate POIs: {e}")
            return []
    def create_prompt_a(
        self, 
        user_id: str, 
        trajectory_id: str, 
        candidateSet: List[Tuple[str, Optional[float], str]],  # Including distance and category
        longterm: List[Tuple[str, str]],  
        recent: List[Tuple[str, str]],    
        time: str
    ) -> str:
        """
        Creates a Chain-of-Thought based prompt for the LLM to recommend the next POI based on
        trajectory history, recent and long-term patterns, and candidate POIs.
        """
        try:
            prompt = f"""\
    Consider the following problem of predicting the next point-of-interest (POI) a user might visit, let’s
think step-by-step:
    
    1. **User's Long-Term Behavior**: The user has shown a long-term preference for certain categories of places. These are their long-term check-ins, formatted as (POIID, Category):
        {longterm}.
        Based on this, think about what types of places the user tends to revisit frequently.
    
    2. **User's Recent Behavior**: Recently, the user has been visiting the following places. These are their recent check-ins, formatted as (POIID, Category):
        {recent}.
        Think about the user's short-term preferences and recent trends in behavior. Are they switching between categories? Is there a pattern to their visits?
    
    3. **Distance Consideration**: Now, consider the candidate POIs that are nearby. These are the candidate POIs, formatted as (POIID, Distance, Category):
        {candidateSet}.
        Think about whether the user is likely to prefer a nearby location, and how the distance might affect their decision.
    
    4. **Sequential Reasoning**: Based on the user's long-term and recent behavior, combined with the distances to candidate POIs, predict the most likely POI the user will visit next. Explain why the user is likely to visit this POI next, considering their preferences and proximity.
    
    Please respond in a valid JSON format containing only the keys: 'recommendation' and 'reason'.
    - "recommendation": A list of the 10 most likely probable POIs (POIIDs) the user might visit next, in descending order of probability.
    - "reason": A step-by-step explanation of your reasoning, covering long-term preferences, recent preferences, and distance.
    """
            return prompt
        except Exception as e:
            self.logger.error(f"Error creating Chain-of-Thought prompt: {e}")
            return ""

    def create_prompt_b(
        self, 
        user_id: str, 
        trajectory_id: str, 
        candidateSet: List[Tuple[str, Optional[float], str]],  
        longterm: List[Tuple[str, str]],  
        recent: List[Tuple[str, str]],    
        time: str
    ) -> str:
        """
        Creates a Plan-and-Solve-based prompt for the LLM to recommend the next POI based on
        trajectory history, recent and long-term patterns, and candidate POIs.
        """
        try:
            prompt = f"""\
        Q: Consider the problem of predicting the next point-of-interest (POI) a user might visit. First, let’s devise a plan to solve this problem.
    
        Plan:
        1. **Identify Long-Term Behavior**: The user has shown a preference for certain types of places over time. These are their long-term check-ins, formatted as (POIID, Category):
            {longterm}.
        We will analyze these to identify patterns in the user's long-term behavior.
    
        2. **Analyze Recent Behavior**: The user has visited the following places recently, formatted as (POIID, Category):
            {recent}.
        We will analyze whether the user's short-term preferences align with or differ from their long-term behavior.
    
        3. **Distance and Proximity Consideration**: Here are the candidate POIs that are nearby, formatted as (POIID, Distance, Category):
            {candidateSet}.
        We will consider the impact of distance on the user's next visit, as users typically prefer closer POIs.
    
        Let's pay special attention to relevant variables, such as POI categories and distances, to ensure we account for all important factors. Calculate intermediate results where necessary, and extract relevant variables and numbers.
    
        Now, let's carry out the plan and predict the next POI the user is likely to visit, step by step.
        Please respond in a valid JSON format containing only the keys: 'recommendation' and 'reason'.
        - "recommendation": A list of the 10 most likely probable POIs (POIIDs) the user might visit next, in descending order of probability.
        - "reason": A step-by-step explanation of your reasoning, covering long-term preferences, recent preferences, and distance.
  
        """
            return prompt
        except Exception as e:
            self.logger.error(f"Error creating Plan-and-Solve prompt: {e}")
            return ""

 
    def create_prompt_c(
        self, 
        user_id: str, 
        trajectory_id: str, 
        poi_id: str, 
        id_range: int, 
        time: str, 
        recent: List[Tuple[str, str]], 
        longterm: List[Tuple[str, str]]
    ) -> str:
        """
        Creates a prompt for the LLM for prompt ID 'c' based on the provided data and trajectory ID.
        """
        try:
            prompt = f"""\
    Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
    <question> The following is a trajectory of user {user_id}: {recent}. \
    There is also historical data: {longterm}. Given the data, at {time}, which POI id \
    will user {user_id} visit? Note that POI id is an identifier in the set of POIs. \
    <answer>: At {time}, user {user_id} will visit POI id {poi_id}.
    Please organize your answer in a JSON object containing the following keys:
    - "recommendation": a list of 10 most likely distinct POIIDs from the candidate set, in descending order of probability.
    """
            return prompt
        except Exception as e:
            self.logger.error(f"Error creating prompt for prompt_id 'c': {e}")
            return ""


    def create_prompt_d(
        self, 
        longterm: List[Tuple[str, str]], 
        recent: List[Tuple[str, str]], 
        candidates: List[Tuple[str, Optional[float], str]]
    ) -> str:
        """
        input --> longterm / recent / candidates   
        Creates a prompt for the LLM based on the user's check-in history and candidate POIs.
        """
        try:
            prompt = f"""\
<long-term check-ins> [Format: (POIID, Category)]: {longterm}
<recent check-ins> [Format: (POIID, Category)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}
Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider the "Distance" since people tend to visit nearby pois.
4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

Please organize your answer in a JSON object containing following keys:
"recommendation" (10 distinct POIIDs of the ten high probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
            return prompt
        except Exception as e:
            self.logger.error(f"Error creating prompt: {e}")
            return ""

    def create_prompt_e(
            self, 
            user_id: str, 
            trajectory_id: str, 
            candidateSet: List[Tuple[str, Optional[float], str]],  # Including distance and category
            longterm: List[Tuple[str, str]],  # Long-term check-ins
            recent: List[Tuple[str, str]],    # Recent check-ins
            time: str
        ) -> str:
            """
            input --> longterm / recent / candidates   
            Creates a prompt for the LLM based on the user's check-in history and candidate POIs.
            """
            try:
                prompt = f"""\
            Consider the following problem of predicting the next point-of-interest (POI) a user might visit, and think through multiple reasoning paths step-by-step:
            
            1. **User's Long-Term Behavior**: The user has shown a long-term preference for certain categories of places. These are their long-term check-ins, formatted as (POIID, Category):
                {longterm}.
                Based on this, generate multiple hypotheses about the types of places the user tends to revisit frequently.
            
            2. **User's Recent Behavior**: Recently, the user has been visiting the following places. These are their recent check-ins, formatted as (POIID, Category):
                {recent}.
                Generate multiple interpretations of the user's short-term preferences and recent trends in behavior. Are they switching between categories? Is there a pattern to their visits?
            
            3. **Distance Consideration**: Now, consider the candidate POIs that are nearby. These are the candidate POIs, formatted as (POIID, Distance, Category):
                {candidateSet}.
                Generate different hypotheses on how the user is likely to prioritize nearby locations based on distance and how the distance might affect their decision.
            
            4. **Sequential Reasoning**: Based on the user's long-term and recent behavior, combined with the distances to candidate POIs, predict the most likely POI the user will visit next by sampling different reasoning paths. Explain the rationale for each path, considering their preferences and proximity.
            
            Pay More Attention! After generating multiple reasoning paths, choose the most consistent prediction across them based on a voting mechanism.
        
            Please respond in a valid JSON format containing only the keys: 'recommendation' and 'reason'.
            - "recommendation": A list of the 10 most likely probable POIs (POIIDs) the user might visit next, in descending order of probability.
            - "reason": A step-by-step explanation of your reasoning, covering long-term preferences, recent preferences, distance, and consistency of the reasoning paths.
            """
                return prompt
            except Exception as e:
                self.logger.error(f"Error creating prompt: {e}")
                return ""

    
    def call_llm_api(self, prompt: str) -> List[str]:
        """
        Calls the appropriate LLM API with the provided prompt and returns the recommendation.
    
        Args:
            prompt (str): The prompt to be sent to the LLM API.
    
        Returns:
            List[str]: A list of recommended POI IDs.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            if self.llm == 'gpt':
                client = OpenAI(
                    api_key=self.key,
                    base_url=self.base
                )
                response = client.chat.completions.create(model='gpt-3.5-turbo', messages=messages, temperature=self.temperature)
            elif self.llm == 'gemini':
                client = OpenAI(api_key=self.key)
                response = client.chat.completions.create(model='gemini-pro', messages=messages, temperature=self.temperature)
            elif self.llm == 'moonshot':
                client = OpenAI(
                    api_key=self.key,
                    base_url=self.base
                )
                response = client.chat.completions.create(model='moonshot-v1-8k', messages=messages, temperature=self.temperature)
            elif self.llm == 'qwen':
                client = OpenAI(
                    api_key=self.key,
                    base_url=self.base
                )
                response = client.chat.completions.create(model='qwen-turbo', messages=messages, temperature=self.temperature)
            elif self.llm == 'claude':
                client = OpenAI(api_key=self.key)
                response = client.chat.completions.create(model='claude-3-5-sonnet-20240620', messages=messages, temperature=self.temperature)
            else:
                raise ValueError(f"Unsupported LLM model: {self.llm}")
    
            content = response.choices[0].message.content
            try:
                # Ensure the content is valid and contains the "recommendation" key
                result = eval(content)
                if isinstance(result, dict) and "recommendation" in result:
                    return result["recommendation"]
                else:
                    self.logger.warning("No 'recommendation' key in LLM response.")
                    return []
            except Exception as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                return []
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            return []

    def get_output_path(self, trajectory: str) -> str:
        """
        Constructs the output path for saving the response based on the trajectory ID.
    
        Args:
            trajectory (str): The trajectory ID.
    
        Returns:
            str: The full file path for saving the response.
        """
        output_directory = '../reflector_output'  
        os.makedirs(output_directory, exist_ok=True) 
        return os.path.join(output_directory, f'response_{trajectory}.json')
    
    def save_response(self, response: Dict, trajectory: str) -> None:
        """
        Saves the response to a JSON file in the output directory.

        Args:
            response (Dict): The response data to save.
            trajectory (str): The trajectory ID for which the response is being saved.
        """
        try:
            path = self.get_output_path(trajectory)
            with open(path, 'w') as file:
                file.write(json.dumps(response, indent=4))
        except Exception as e:
            self.logger.error(f"Error saving response: {e}")

    def create_output_data(
        self, 
        prompt: str, 
        prediction: List[str], 
        groundTruth: List[str]
        ) -> Dict[str, Optional[str]]:
        """
        Creates the output data to be saved, including the prompt, response, and ground truth.

        Args:
            prompt (str): The prompt used for the LLM.
            prediction (List[str]): The list of predicted POI IDs.
            groundTruth (List[str]): The ground truth POI IDs.

        Returns:
            Dict[str, Optional[str]]: The output data dictionary.
        """
        try:
            output = {
                "prompt": prompt,
                "response": {"recommendation": prediction},
                "groundTruth": groundTruth[0] if groundTruth else None
            }
            return output
        except Exception as e:
            self.logger.error(f"Error creating output data: {e}")
            return {}
