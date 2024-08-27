import pandas as pd
import os
import numpy as np
from tqdm import tqdm 
class DataAgent:
    def __init__(self, datasetName, case_num, filePath=None):
        self.datasetName = datasetName
        self.case_num = case_num
        self.filePath = filePath
        
        if datasetName in ['nyc', 'tky', 'ca']:
            self.trainPath, self.testPath = self.setPaths()
            if datasetName in ['tky', 'ca']:
                self.split_data()
            data = self.getData()
            self.longs = data["longs"]
            self.recents = data["recents"]
            self.targets = data["targets"]
            self.poiInfos = data["poiInfos"]
            self.traj2u = data["traj2u"]
            self.poiList = data["poiList"]
        else:
            raise ValueError(f"Unsupported dataset: {datasetName}")
    
    def setPaths(self):
        if self.datasetName == 'nyc':
            self.filePath = './data/nyc/raw/NYC_{}.csv'
        elif self.datasetName == 'tky':
            self.filePath = './data/tky/raw/dataset_TSMC2014_TKY.txt'
        elif self.datasetName == 'ca':
            self.filePath = './data/ca/raw/dataset_gowalla_ca_ne.csv'
        else:
            raise NotImplementedError(f"Dataset {self.datasetName} is not implemented.")
        
        trainPath = os.path.join(os.path.dirname(self.filePath), f'{self.datasetName}_train.csv')
        # validPath = os.path.join(os.path.dirname(self.filePath), f'{self.datasetName}_valid.csv')
        testPath = os.path.join(os.path.dirname(self.filePath), f'{self.datasetName}_test.csv')
        return trainPath, testPath 


    def split_data(self):
        print(f"Reading data from {self.filePath} and splitting it into train, validation, and test sets.")
        """
        if self.datasetName == 'tky':
            data = pd.read_csv(self.filePath, sep='\t', header=None, encoding='ISO-8859-1')
            print("Data type:", type(data))  # excepection: <class 'pandas.core.frame.DataFrame'>
            data.columns = ['user_id', 'POI_id','POI_catid','POI_catname','latitude','longitude','timezone','UTC_time']
        """
        if self.datasetName == 'tky':
            data = pd.read_csv(self.filePath, sep='\t', header=None, encoding='ISO-8859-1')
            print("Data type:", type(data))  # Expected: <class 'pandas.core.frame.DataFrame'>
            data.columns = ['user_id', 'POI_id', 'POI_catid', 'POI_catname', 'latitude', 'longitude', 'timezone', 'UTC_time']
            
            # Sort data by user_id and UTC_time to generate the correct sequence
            data = data.sort_values(by=['user_id', 'UTC_time'])

            # Generate trajectory_id as user_id + visit sequence number
            data['trajectory_id'] = data.groupby('user_id').cumcount() + 1
            data['trajectory_id'] = data['user_id'].astype(str) + '_' + data['trajectory_id'].astype(str)

        elif self.datasetName == 'ca':
            data = pd.read_csv(self.filePath, encoding='ISO-8859-1')
        else:
            raise NotImplementedError(f"Dataset {self.datasetName} is not implemented.")

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data
        train_split = int(0.8 * len(data))
        valid_split = int(0.9 * len(data))

        train_data = data.iloc[:train_split]
        valid_data = data.iloc[train_split:valid_split]
        test_data = data.iloc[valid_split:]

        train_data.to_csv(self.trainPath, index=False)
        valid_data.to_csv(self.testPath.replace('_test.csv', '_valid.csv'), index=False)
        test_data.to_csv(self.testPath, index=False)

        print(f"Data split completed. Train: {len(train_data)}, Validation: {len(valid_data)}, Test: {len(test_data)}.")


    def readTrain(self):
        if not os.path.exists(self.trainPath):
            raise FileNotFoundError(f"Training data file not found: {self.trainPath}")
        print(f"Trying to open training data file: {self.trainPath}")
        longs = dict()
        pois = dict()
        with open(self.trainPath, 'r') as file:
            lines = file.readlines()
        for line in lines[1:]:
            data = line.strip().split(',')
            
            if self.datasetName == 'nyc':
                # time, u, lati, longi, i, category = data[1], data[5], data[6], data[7], data[8], data[10]
                # print("time: {}, u: {}, lati: {}, longi: {}, i: {}, category: {}".format(data[1],data[5], data[6], data[7], data[8], data[10]))
                time, u, lati, longi, i, category = data[8], data[0], data[5], data[6], data[2], data[4]
            elif self.datasetName == 'tky':
                if len(data) < 8:
                    print(f"Skipping line due to insufficient data: {data}")
                    continue
                try:
                    time, u, lati, longi, i, category = data[7], data[0], data[5], data[6], data[2], data[3]
                    # print("time: {}, u: {}, lati: {}, longi: {}, i: {}, category: {}".format(data[7], data[0], data[5], data[6], data[2], data[3]))
                
                except IndexError as e:
                    print(f"Index error encountered: {e}, line data: {data}")
                    continue

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
            data = line.strip().split(',')
            
            if self.datasetName == 'nyc':
                # time, trajectory, u, lati, longi, i, category = data[1], data[3], data[5], data[6], data[7], data[8], data[10]
                time, trajectory, u, lati, longi, i, category = data[8], data[3], data[0], data[5], data[6], data[2], data[4]
            elif self.datasetName == 'tky':
                if len(data) < 9:
                    print(f"Skipping line due to insufficient data: {data}")
                    continue
                try:
                    time, trajectory, u, lati, longi, i, category = data[7], data[8], data[0], data[5], data[6], data[2], data[3]
                except IndexError as e:
                    print(f"Index error encountered: {e}, line data: {data}")
                    continue

            if i not in pois:
                pois[i] = {"latitude": lati, "longitude": longi, "category": category}
            if trajectory not in traj2u:
                traj2u[trajectory] = u
            if trajectory not in recents:
                recents[trajectory] = list()
            recents[trajectory].append((i, time))
            targets[trajectory] = (i, time)
        
        return recents, pois, targets, traj2u
    
    def getData(self):
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

        poiInfos.update(testPoi)
        targets = dict(list(targets.items())[:self.case_num])

        return {
            "longs": longs,
            "recents": recents,
            "targets": targets,
            "poiInfos": poiInfos,
            "traj2u": traj2u,
            "poiList": list(poiInfos.keys())
        }

    def process(self):
        if self.datasetName in ['nyc', 'tky']:
            return self.getData()
        elif self.datasetName == 'ca':
            return self.ca_process()
        else:
            raise ValueError(f"Unsupported dataset: {self.datasetName}")
        

### keyerror 是由于latitude生成了错误的值，无法查询  40.71981038                 KeyError('40.67208907715739')
# Workflow completed. ACC@1: 0.7900, ACC@5: 0.8300, ACC@10: 0.8300, MRR: 0.8070
# Trajectory type: <class 'str'>, value: 39    KeyError('40.67389763442464')    
# # 40.72227958 40.723206 40.723955 40.66390777 40.7244943 40.72939547  40.723206

# 40.7431989535266  40.7536956621939




