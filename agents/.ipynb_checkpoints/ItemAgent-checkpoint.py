import os
import json
from openai import OpenAI
from math import radians, sin, cos, sqrt, atan2
from zhipuai import ZhipuAI
from tools.API import Spark, ERNIE, Llama

"""
DataAgent  --> 输出数据 --> 2. ManageAgent  --> 分配任务 --> 3. ItemAgent  --> 返回推荐结果 --> 4. ManageAgent  
--> 汇总结果 --> 5. ReflectorAgent  --> 输出最终结果
"""




class ItemAgent:
    def __init__(self, llm, logger, temperature):
        self.logger = logger
        self.llm = self.set_llm(llm)
        self.temperature = temperature
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
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1 = eval(lat1)
        lon1 = eval(lon1)
        lat2 = eval(lat2)
        lon2 = eval(lon2)
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        radius = 6371.0
        distance = radius * c
        return distance


    def generate_recommendation(self, trajectory, candidateSet, groundTruth, longs, recents, traj2u, poiInfos):
        u = traj2u[trajectory]
        long = longs[u]
        rec = recents[trajectory]

        # print("Item Agent Begining...", flush=True)
        base_dir = os.path.dirname(os.path.abspath(__file__))


        path = os.path.join(base_dir, 'output', 'LLMWorkflow', trajectory)
        os.makedirs(os.path.dirname(path), exist_ok=True)


        if os.path.exists(path):
            with open(path, 'r') as file:
                response = file.read()
                res_content = eval(response)
                prediction = res_content["response"]["recommendation"]
        else:
            output = dict()
            mostrec = rec[-1][0]
            longterm = [(poi, poiInfos[poi]["category"]) for poi, _ in long][-40:]
            recent = [(poi, poiInfos[poi]["category"]) for poi, _ in rec][-5:]
            candidates = [
                (poi, self.haversine_distance(poiInfos[poi]["latitude"], poiInfos[poi]["longitude"],
                                         poiInfos[mostrec]["latitude"], poiInfos[mostrec]["longitude"]),
                 poiInfos[poi]["category"])
                for poi in candidateSet
            ]
            candidates.sort(key=lambda x: x[1])

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
"recommendation" (10 distinct POIIDs of the ten most probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
            prompt1 = f"""\
<long-term check-ins> [Format: (POIID, Category)]: {longterm}
<recent check-ins> [Format: (POIID, Category)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}

Context:
The following data represents a user's check-in history at various points of interest (POIs). The long-term check-ins list the POIs the user has frequently visited over a longer period, reflecting their enduring preferences. The recent check-ins highlight the most recent POIs visited by the user, indicating their current interests and possibly shifting preferences. The candidate set includes potential next POIs the user might visit, with each candidate annotated with its distance from the last visited POI and its category.

Objective:
Based on the user's long-term and recent check-ins, as well as the candidate set, your task is to recommend the next POI the user is most likely to visit. When making your recommendation, consider the following factors:
1. **Long-term Preferences**: Users often revisit places they have frequently visited in the past.
2. **Recent Preferences**: The most recent visits may indicate a temporary shift or trend in the user's preferences.
3. **Proximity**: Users tend to favor nearby locations, so distance plays a crucial role in predicting the next POI.
4. **Category Transition**: Consider the likelihood of the user transitioning from one category to another based on their historical behavior.

Task:
Recommend the 10 most probable POIs from the candidate set that the user is likely to visit next. Provide your recommendation in a JSON format with two keys:
1. **recommendation**: An ordered list of 10 POIIDs (from the candidate set) that the user is most likely to visit, listed in descending order of probability.
2. **reason**: A concise explanation that supports your recommendation, referring to the user's long-term preferences, recent preferences, distance, and category transition as relevant.

Please ensure the response is concise and focused, with no line breaks.
"""
            output["prompt"] = prompt
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
            

            res_content = eval(response.choices[0].message.content)
            output["response"] = res_content
            prediction = res_content["recommendation"]
            output["groundTruth"] = groundTruth[0]
            self.outputResponse(output, trajectory)
        return prediction

    def outputResponse(self, response, trajectory):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        path = os.path.join(parent_dir, 'output', 'LLMWorkflow', self.llm, trajectory)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            file.write(json.dumps(response, indent='\t'))