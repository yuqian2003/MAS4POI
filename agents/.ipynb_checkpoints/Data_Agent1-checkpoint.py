# This agent is used to preprocess POI data
import os.path as osp
import pandas as pd
import json
import shapely
import os



class DataAgent:
    def __init__(self, datasetName, case_num, filePath = None):
        """
        self.datasetName = datasetName
        self.filePath = filePath
        self.case_num = case_num
        self.trainPath, self.testPath = self.setPaths()
        data = self.getData()  # 获取数据字典
        self.longs = data["longs"]
        self.recents = data["recents"]
        self.targets = data["targets"]
        self.poiInfos = data["poiInfos"]
        self.traj2u = data["traj2u"]
        self.poiList = data["poiList"]
        """
        self.datasetName = datasetName
        self.case_num = case_num
        if datasetName in ['nyc', 'tky']:
            self.case_num = case_num
            self.filePath = filePath
            self.trainPath, self.testPath = self.setPaths()
            data = self.getData()  # 获取数据字典
            self.longs = data["longs"]
            self.recents = data["recents"]
            self.targets = data["targets"]
            self.poiInfos = data["poiInfos"]
            self.traj2u = data["traj2u"]
            self.poiList = data["poiList"]
        elif datasetName == 'ca':
            self.rootDir = self.get_root_dir()
            self.case_num = case_num
            self.filePath = filePath
    
    def setPaths(self):
        if self.datasetName == 'nyc':
            self.filePath = './data/nyc/raw/nyc_{}.csv'
            # self.filePath = '/home/yuqian.wu/Megatron-LM/zPOI/LLMMove/models/data/nyc/raw/NYC_{}.csv'
        elif self.datasetName == 'tky':
            self.filePath = './data/tky/raw/dataset_TSMC2014_TKY.txt'  # dataset_TSMC2014_TKY.txt
        elif self.datasetName == 'ca':
            self.filePath = '.data/ca/raw/dataset_gowalla_ca_ne.csv'
        else:
            raise NotImplementedError(f"Dataset {self.datasetName} is not implemented.")

        trainPath = self.filePath.format('train')
        testPath = self.filePath.format('test')
        print(f"Training data path: {trainPath}")
        print(f"Test data path: {testPath}")
        return trainPath, testPath

    def readTrain(self):
        if not os.path.exists(self.trainPath):
            raise FileNotFoundError(f"Training data file not found: {self.trainPath}")
        print(f"Trying to open training data file: {self.trainPath}")
        longs = dict()
        pois = dict()
        with open(self.trainPath, 'r') as file:
            # print(file.readline())
            lines = file.readlines()
        for line in lines[1:]:
            data = line.split(',')
            time, u, lati, longi, i, category = data[1], data[5], data[6], data[7], data[8], data[10]
            if i not in pois:
                pois[i] = {"latitude": lati, "longitude": longi, "category": category}
            if u not in longs:
                longs[u] = list()
            longs[u].append((i, time))
        return longs, pois

    def readTest(self):
        if not os.path.exists(self.testPath):
            raise FileNotFoundError(f"Test data file not found: {self.testPath}")
        recents = dict()
        pois = dict()
        targets = dict()
        traj2u = dict()
        with open(self.testPath, 'r') as file:
            lines = file.readlines()
        for line in lines[1:]:
            data = line.split(',')
            time, trajectory, u, lati, longi, i, category = data[1], data[3], data[5], data[6], data[7], data[8], data[
                10]
            if i not in pois:
                pois[i] = dict()
                pois[i]["latitude"] = lati
                pois[i]["longitude"] = longi
                pois[i]["category"] = category
            if trajectory not in traj2u:
                traj2u[trajectory] = u
            if trajectory not in recents:
                recents[trajectory] = list()
                recents[trajectory].append((i, time))
            else:
                if trajectory in targets:
                    recents[trajectory].append(targets[trajectory])
                targets[trajectory] = (i, time)
        return recents, pois, targets, traj2u

    def getData(self):
        # Paths are already set, no need to set again
        try:
            longs, poiInfos = self.readTrain()
        except Exception as e:
            print("The error is caused by:", e)
            raise

        try:
            recents, testPoi, targets, traj2u = self.readTest()
        except Exception as e:
            print("The error is caused by:", e)
            raise

        # Update POI info with test data
        poiInfos.update(testPoi)

        # Limit number of targets
        targets = dict(list(targets.items())[:self.case_num])

        # 返回值作为一个字典
        return {
            "longs": longs,
            "recents": recents,
            "targets": targets,
            "poiInfos": poiInfos,
            "traj2u": traj2u,
            "poiList": list(poiInfos.keys())
        }

    def get_root_dir(self):
        dirname = os.getcwd()
        dirname_split = dirname.split("/")
        index = dirname_split.index("preprocessing")
        dirname = "/".join(dirname_split[:index + 1])
        return dirname

    def ca_process(self):
        data_path = osp.join(self.rootDir, 'data', 'ca', 'raw')
        raw_checkins = pd.read_csv(osp.join(data_path, 'loc-gowalla_totalCheckins.txt'), sep='\t', header=None)
        raw_checkins.columns = ['userid', 'datetime', 'checkins_lat', 'checkins_lng', 'id']
        subset1 = pd.read_csv(osp.join(data_path, 'gowalla_spots_subset1.csv'))
        raw_checkins_subset1 = raw_checkins.merge(subset1, on='id')
        with open(osp.join(data_path, 'us_state_polygon_json.json'), 'r') as f:
            us_state_polygon = json.load(f)
        for i in us_state_polygon['features']:
            if i['properties']['name'].lower() == 'california':
                california = shapely.geometry.Polygon(i['geometry']['coordinates'][0])
            if i['properties']['name'].lower() == 'nevada':
                nevada = shapely.geometry.Polygon(i['geometry']['coordinates'][0])

        raw_checkins_subset1['is_ca'] = raw_checkins_subset1.apply(
            lambda x: nevada.intersects(
                shapely.geometry.Point(x['checkins_lng'], x['checkins_lat'])) or california.intersects(
                shapely.geometry.Point(x['checkins_lng'], x['checkins_lat'])), axis=1
        )
        raw_checkins_subset1 = raw_checkins_subset1[raw_checkins_subset1['is_ca']]
        df = raw_checkins_subset1[['userid', 'id', 'spot_categories', 'checkins_lat', 'checkins_lng', 'datetime']]
        df.columns = ['UserId', 'PoiId', 'PoiCategoryId', 'Latitude', 'Longitude', 'UTCTime']
        df.to_csv(osp.join(data_path, 'dataset_gowalla_ca_ne.csv'), index=False)

    def process(self):
        if self.datasetName in ['nyc', 'tky']:
            return self.getData()
        elif self.datasetName == 'ca':
            return self.ca_process()
        else:
            raise ValueError(f"Unsupported dataset: {self.datasetName}")
