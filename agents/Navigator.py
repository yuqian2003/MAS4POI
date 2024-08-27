import json
import math
from openai import OpenAI
from zhipuai import ZhipuAI
from urllib.parse import quote
import urllib.request as request
from typing import List, Dict, Union, Any, Tuple
from langchain.prompts import PromptTemplate

class NaviAgent:
    def __init__(self, amap_api_key : str, source: Union[str, tuple], target: Union[str, tuple], 
                 city: str, llm, logger, temperature, key: str, base: str,
                 use_coordinates: bool = False) -> None:
        """
        Initializes the NaviAgent with the given parameters.
        
        Parameters:
        - api_key (str): API key for the map services.
        - source (Union[str, tuple]): Source location (address as a string or coordinates as a tuple of (lat, lon)).
        - target (Union[str, tuple]): Target location (address as a string or coordinates as a tuple of (lat, lon)).
        - city (str): City name for address-to-coordinate conversion (ignored if coordinates are used directly).
        - llm (str): The name of the LLM to use (e.g., 'gpt', 'claude').
        - logger (Logger): Logger for tracking actions.
        - temperature (float): Temperature setting for LLM.
        - use_coordinates (bool): If True, use coordinates directly; if False, convert addresses to coordinates.
        """
        self.history = []
        self.amap_api_key  = amap_api_key 
        self.target = target
        self.source = source
        self.city = city
        self.llm = None
        self.logger = logger
        self.temperature = temperature
        self.use_coordinates = use_coordinates  
        self.llm = self.set_llm(llm)
        # OPENAI
        self.key = key
        self.base = base
    def set_llm(self, llm_name):
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
            self.logger.info(f"Navigator starting {llm_models[llm_name]}...")
            return llm_name
        else:
            self.logger.error(f"Unknown LLM model: {llm_name}. Defaulting to GPT model.")
            return 'gpt'

    def address_to_coordinates(self, address: str) -> str:
        base_url = "https://restapi.amap.com/v3/geocode/geo"
        encoded_address = quote(address)
        encoded_city = quote(self.city)
        params = f"?key={self.amap_api_key }&address={encoded_address}&city={encoded_city}"
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
        if self.use_coordinates:
            # If using coordinates directly
            source = f"{self.source[1]},{self.source[0]}" 
            target = f"{self.target[1]},{self.target[0]}"  
        else:
            # Convert addresses to coordinates
            source = self.address_to_coordinates(self.source)
            target = self.address_to_coordinates(self.target)
        
        base_url = "https://restapi.amap.com/v5/direction/walking"   #### WALK!
        if self.use_coordinates == True:
            params = f"?key={self.amap_api_key}&isindoor=1&origin={self.source}&destination={self.target}&alternative_route=2&show_fields=cost,polyline"
        else:
            params = f"?key={self.amap_api_key}&isindoor=1&origin={source}&destination={target}&alternative_route=2&show_fields=cost,polyline"
        url = base_url + params
        html = request.urlopen(url).read()
        js = json.loads(html)
        return js

    def get_weather(self) -> Dict:
        base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        encoded_city = quote(self.city)
        params = f"?city={encoded_city}&key={self.amap_api_key }"
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

    def haversine(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculates the Haversine distance between two geographic points on the Earth's surface.

        The Haversine formula calculates the great-circle distance between two points, which is the shortest distance over the Earth's surface.

        Parameters:
        - coord1 (Tuple[float, float]): The latitude and longitude of the first point (in degrees).
        - coord2 (Tuple[float, float]): The latitude and longitude of the second point (in degrees).

        Returns:
        - float: The distance between the two points in kilometers.

        Formula:
        - a = sin²(Δφ / 2) + cos(φ1) * cos(φ2) * sin²(Δλ / 2)
        - c = 2 * atan2(√a, √(1−a))
        - distance = R * c
        Where:
        - φ is latitude, λ is longitude
        - R is the Earth's radius (mean radius = 6,371 km)
        """
        try:
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            R = 6371  # Earth's radius in kilometers

            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)

            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance = R * c
            return distance
        except Exception as e:
            self.logger.error(f"Error calculating Haversine distance: {e}")
            return 0.0

    def generate_static_map_url(self, polyline_points: List[str]) -> str:
        if self.use_coordinates:
            source = f"{self.source[1]},{self.source[0]}"
            target = f"{self.target[1]},{self.target[0]}"
        else:
            source = self.address_to_coordinates(self.source)
            target = self.address_to_coordinates(self.target)
        
        bounds = self.calculate_bounds(polyline_points)
        center, zoom_level = self.calculate_center_and_zoom(bounds)
        paths_param = self.build_paths_param(polyline_points)

        static_map_url = (
            f"https://restapi.amap.com/v3/staticmap?location={center}&zoom={zoom_level}&size=1000*1000&scale=2"
            f"&markers=mid,0xFF0000,A:{source}|mid,0xFF0000,B:{target}"  
            f"&labels=Source,1,0,16,0xFF0000,0xFFFFFF:{source}|Target,2,0,16,0xFF0000,0xFFFFFF:{target}"
            f"&paths={paths_param}&key={self.amap_api_key }"
        )
        print("You can click this link to view your map:", static_map_url)
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
        # if self.use_coordinates == True:
        instructions = self.instructions_js(route_data)
        instructions_str = "\n".join(instructions)

        polyline_points = self.extract_polyline_points1(route_data['route']['paths'])
        static_map_url = self.generate_static_map_url(polyline_points)

        path = "./map_generate/map_from_SOURCE_to_TARGET.png"

        save_path = path.format(self.source, self.target)
        self.download_map_image(static_map_url, save_path)
        
        prompt_template = (
            "You are an intelligent navigation assistant.\n"
            "You excel at providing accurate and clear navigation instructions.\n"
            "Your task is to plan a route.\n"
            "The objective is to assist the user who needs navigation help.\n"
            f"The task scope is to plan a route from the specified starting point {self.source} to the destination {self.target}. You need to provide recommendations based on the following instructions: {instructions_str}\n"
            "The task success criterion is determining the best walking route.\n"
            "The task constraints are that the route must be limited to walking, and outdoor navigation should be considered.\n"
            "The output format should be a step-by-step navigation guide.\n"
            "The expected output is a detailed list of navigation steps.\n"
        )
        template = PromptTemplate(template=prompt_template, input_variables=["source", "target", "instructions_str"])
        prompt = template.format(source=self.source, target=self.target, instructions_str=instructions_str)
        messages = [{"role": "user", "content": prompt}]
        self.history.append({"role": "user", "content": prompt})
        
        try:
            messages = [{"role": "user", "content": prompt}]

            if self.llm == 'gpt':
                client = OpenAI(
                    api_key  = self.key,
                    base_url = self.base
                )
                response = client.chat.completions.create(model='gpt-3.5-turbo', messages=messages, temperature=self.temperature)
            elif self.llm == 'gemini':
                client = OpenAI(api_key=self.key)
                response = client.chat.completions.create(model='gemini-pro', messages=messages, temperature=self.temperature)
            elif self.llm == 'moonshot':
                client = OpenAI(api_key=self.key)
                response = client.chat.completions.create(model='moonshot-v1-8k', messages=messages, temperature=self.temperature)
            elif self.llm == 'qwen':
                client = OpenAI(api_key=self.key)
                response = client.chat.completions.create(model='qwen-turbo', messages=messages, temperature=self.temperature)
            elif self.llm == 'claude':
                client = OpenAI(api_key=self.key)
                response = client.chat.completions.create(model='claude-3-5-sonnet-20240620', messages=messages, temperature=self.temperature)
            else:
                raise ValueError(f"Unsupported LLM model: {self.llm}")
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            return []

        try:
            answer = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}, treating response as plain text.")
            answer = response.choices[0].message.content
        self.logger.info(answer)
        self.history.append({"role": "assistant", "content": answer})
        # print("This is the instructions_str: \n" + instructions_str)
        # print("\n This is the static map URL: \n", static_map_url)
        # self.logger.info("This is the history: ", self.history)
        return answer