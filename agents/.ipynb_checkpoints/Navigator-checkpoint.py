import json
import math
import urllib.request as request
from tools.API import Spark
from langchain.prompts import PromptTemplate
from typing import List, Dict
from urllib.parse import quote
from zhipuai import ZhipuAI
from tools.API import Spark, ERNIE, Llama

class NaviAgent:
    def __init__(self, api_key: str, source: str, target: str, city: str, llm, logger, temperature) -> None:
        self.history = []
        self.api_key = api_key
        self.target = target
        self.source = source
        self.city = city
        self.llm = self.set_llm(llm)
        self.logger = logger
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
    def address_to_coordinates(self, address: str) -> str:
        base_url = "https://restapi.amap.com/v3/geocode/geo"
        encoded_address = quote(address)
        encoded_city = quote(self.city)
        params = f"?key={self.api_key}&address={encoded_address}&city={encoded_city}"
        url = base_url + params
        try:
            response = request.urlopen(url)
            data = json.loads(response.read())
            if data['status'] == '1' and data['geocodes']:
                location = data['geocodes'][0]['location']
                return location
            else:
                raise Exception(f"Failed to convert address to coordinates: {data['info']}")
        except Exception as e:
            print(f"Error during address conversion: {e}")
            return ""

    def get_route_url(self) -> Dict:
        source = self.address_to_coordinates(self.source)
        target = self.address_to_coordinates(self.target)
        base_url = "https://restapi.amap.com/v5/direction/walking"
        params = f"?key={self.api_key}&isindoor=1&origin={source}&destination={target}&alternative_route=2&show_fields=cost,polyline"
        url = base_url + params
        html = request.urlopen(url).read()
        js = json.loads(html)
        return js

    def get_weather(self) -> Dict:
        base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        encoded_city = quote(self.city)
        params = f"?city={encoded_city}&key={self.api_key}"
        url = base_url + params
        try:
            response = request.urlopen(url)
            data = json.loads(response.read())
            if data['status'] == '1' and data['lives']:
                weather_info = data['lives'][0]
                weather_data = {
                    "weather": weather_info['weather'],
                    "temperature": weather_info['temperature'],
                    "winddirection": weather_info['winddirection'],
                    "windpower": weather_info['windpower'],
                    "reporttime": weather_info['reporttime']
                }
                print(json.dumps(weather_data, ensure_ascii=False, indent=4))
                return weather_data
            else:
                raise Exception(f"Failed to fetch weather information: {data['info']}")
        except Exception as e:
            print(f"Error during weather fetching: {e}")
            return ""

    def instructions_js(self, js: Dict) -> List[str]:
        p = js['route']['paths'][0]
        instructions = [step['instruction'] for step in p['steps']]
        return instructions

    def extract_polyline_points(self, paths: List[Dict]) -> List[str]:
        polylines = []
        for path in paths:
            for step in path['steps']:
                polylines.extend(step['polyline'].split(';'))
        return polylines

    def extract_polyline_points1(self, paths: List[Dict]) -> List[str]:
        polyline_points = []
        for step in paths[0]['steps']:
            points = step['polyline'].split(';')
            polyline_points.extend(points)
        seen = set()
        unique_points = []
        for point in polyline_points:
            if point not in seen:
                unique_points.append(point)
                seen.add(point)
        return unique_points


    def build_paths_param(self, polyline_points: List[str]) -> str:
        path_param = '8,0x4682B4,0.8,,0.5:'
        path_param += ';'.join(polyline_points)

        return path_param

    def calculate_bounds(self, polyline_points: List[str]) -> Dict[str, float]:
        lats = [float(point.split(',')[1]) for point in polyline_points]
        lons = [float(point.split(',')[0]) for point in polyline_points]
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
        }

    def calculate_center_and_zoom(self, bounds: Dict[str, float]) -> (str, int):
        center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
        center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
        max_distance = max(
            self.haversine((bounds['min_lat'], bounds['min_lon']), (bounds['max_lat'], bounds['max_lon'])),
            self.haversine((bounds['min_lat'], bounds['max_lon']), (bounds['max_lat'], bounds['min_lon']))
        )
        
        # Adjust zoom level based on max_distance to fit the entire route
        if max_distance > 100:
            zoom_level = 8
        elif max_distance > 50:
            zoom_level = 10
        elif max_distance > 20:
            zoom_level = 12
        elif max_distance > 10:
            zoom_level = 13
        else:
            zoom_level = 15
    
        return f"{center_lon},{center_lat}", zoom_level


    def haversine(self, coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371  # 地球半径，单位为公里
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def generate_static_map_url(self, polyline_points: List[str]) -> str:
        source = self.address_to_coordinates(self.source)
        target = self.address_to_coordinates(self.target)
        bounds = self.calculate_bounds(polyline_points)
        center, zoom_level = self.calculate_center_and_zoom(bounds)
        paths_param = self.build_paths_param(polyline_points)
        static_map_url = (
            f"https://restapi.amap.com/v3/staticmap?location={center}&zoom={zoom_level}&size=1000*1000&scale=2"
            f"&markers=mid,0xFF0000,A:{source}|mid,0xFF0000,B:{target}"  # 修改标记颜色为红色
            f"&labels=Source,1,0,16,0xFF0000,0xFFFFFF:{source}|Target,2,0,16,0xFF0000,0xFFFFFF:{target}"
            f"&paths={paths_param}&key={self.api_key}"
        )
        print("You can clik this link to view your map:", static_map_url)
        return static_map_url

    def download_map_image(self, url, save_path):
        try:
            with request.urlopen(url) as response, open(save_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print(f"Map image saved to {save_path}")
        except Exception as e:
            print(f"Failed to download map image: {e}")

    def answer(self) -> str:
        route_data = self.get_route_url()
        instructions = self.instructions_js(route_data)
        instructions_str = "\n".join(instructions)
        polyline_points = self.extract_polyline_points1(route_data['route']['paths'])
        static_map_url = self.generate_static_map_url(polyline_points)
        path = "./map_generate/map_from_{}_to_{}.png"
        save_path = path.format(self.source,self.target)
        self.download_map_image(static_map_url, save_path)
        prompt_template = (
            "你是一个智能导航助手。\n"
            "你擅长提供准确而清晰的导航指引。\n"
            "任务是进行路线规划。\n"
            "目标是需要导航协助的用户。\n"
            # "任务背景是在中国广东省清远市洲心街道的轮洲岛上进行导航。\n"
            f"任务范围是从指定的出发地 {self.source} 到目的地 {self.target},你需要根据{instructions_str}来给出建议\n"
            "任务解决判定标准是确定最佳步行路线。\n"
            "任务限制条件是仅限步行路线，并考虑室外导航。\n"
            "输出格式是提供逐步导航指引。\n"
            "输出量是一份详细的导航步骤列表。\n"
        )
        template = PromptTemplate(template=prompt_template, input_variables=["source", "target", "instructions_str"])
        prompt = template.format(source=self.source, target=self.target, instructions_str=instructions_str)
        messages=[{"role": "user", "content": prompt}]
        self.history.append({"role": "user", "content": prompt})
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
            

        answer = eval(response.choices[0].message.content)
        self.history.append({"role": "assistant", "content": answer})
        print("This is the instructions_str: \n" + instructions_str)
        print("\n This is the 静态地图URL: \n", static_map_url)
        print("This is the history: ", self.history)
        return answer

api_key = "d2f26e63f57b19fb165a9371f8329fbf"
source = "山东省青岛市城阳区王沙路36号"
target = "山东省青岛市城阳区玉虹路16号"
city = "青岛"
