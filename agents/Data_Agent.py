import os
import pandas as pd
import numpy as np
from tqdm import tqdm 
from typing import List, Dict, Tuple, Optional, Any
class DataAgent:
    
    """
    DataAgent is a data processing class that loads and processes location-based datasets
    (NYC, Tokyo) for trajectory-based predictions.

    Parameters:
    datasetName (str): The name of the dataset (nyc, tky, ca).
    case_num (int): The number of test cases to load.
    group (str): User group, such as 'very_active', 'normal', 'inactive', or 'overall'.
    filePath (str, optional): Custom file path for the dataset files.
    """
    
    def __init__(self, datasetName: str, case_num: int, group: str, filePath: Optional[str] = None) -> None:

        """
        Initializes the DataAgent object and sets up file paths, then loads and processes data.

        Parameters:
        - datasetName (str): Name of the dataset.
        - case_num (int): Number of test cases to load.
        - group (str): User group.
        - filePath (str, optional): Custom file path for the dataset files.
        
        Initializes the following attributes:
        - self.longs: Dictionary storing long-term user trajectory data.
        - self.recents: Dictionary storing recent user trajectory data.
        - self.targets: Dictionary storing target points of interest (POIs).
        - self.poiInfos: Dictionary storing POI information (latitude, longitude, category).
        - self.traj2u: Dictionary mapping trajectory IDs to user IDs.
        - self.poiList: List of unique POIs.
        """

        self.datasetName = datasetName
        self.case_num = case_num
        self.filePath = filePath
        self.group = group

        if datasetName in ['nyc', 'tky']:
            self.trainPath, self.testPath = self.setPaths()
            data = self.getData()
            self.longs = data["longs"]
            self.recents = data["recents"]
            self.targets = data["targets"]
            self.poiInfos = data["poiInfos"]
            self.traj2u = data["traj2u"]
            self.poiList = data["poiList"]
        else:
            raise ValueError(f"Unsupported dataset: {datasetName}")
    
    def setPaths(self) -> Tuple[str, str]:

        """
        Sets the paths for the training and testing data files based on the dataset name and group.

        Returns:
        - trainPath (str): Path to the training data file.
        - testPath (str): Path to the testing data file.
        
        Raises:
        - NotImplementedError: If the dataset name is not recognized.
        """

        if self.datasetName == 'nyc':
            if self.group == 'overall':
                self.filePath = './data/nyc/raw/NYC_{}.csv'
            else:
                self.filePath = './data/nyc/raw/{}/NYC_{{}}.csv'.format(self.group)
                # 'very_active','normal','inactive','overall'
        elif self.datasetName == 'tky':
            # self.filePath = './data/tky/raw/dataset_TSMC2014_TKY.txt'
            if self.group == 'overall':
                self.filePath = './data/tky/raw/tky_{}.csv'
            else:
                self.filePath = './data/tky/raw/{}/tky_{{}}.csv'.format(self.group)
        else:
            raise NotImplementedError(f"Dataset {self.datasetName} is not implemented.")
        
        trainPath = os.path.join(os.path.dirname(self.filePath), f'{self.datasetName}_train.csv')
        # validPath = os.path.join(os.path.dirname(self.filePath), f'{self.datasetName}_valid.csv')
        testPath = os.path.join(os.path.dirname(self.filePath), f'{self.datasetName}_test.csv')
        return trainPath, testPath 


    def split_data(self) -> None:
                
        """
        Splits the raw data into training, validation, and test sets and saves them to CSV files.
        This function is primarily used for the Tokyo dataset ('tky').

        Raises:
        - NotImplementedError: If the dataset is not 'tky' or 'ca'.
        """

        print(f"Reading data from {self.filePath} and splitting it into train, validation, and test sets.")
        if self.datasetName == 'tky':
            data = pd.read_csv(self.filePath, sep='\t', header=None, encoding='ISO-8859-1')
            # print("Data type:", type(data))  # Expected: <class 'pandas.core.frame.DataFrame'>
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

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)  
        train_split = int(0.8 * len(data))
        valid_split = int(0.9 * len(data))

        train_data = data.iloc[:train_split]
        valid_data = data.iloc[train_split:valid_split]
        test_data = data.iloc[valid_split:]

        train_data.to_csv(self.trainPath, index=False)
        valid_data.to_csv(self.testPath.replace('_test.csv', '_valid.csv'), index=False)
        test_data.to_csv(self.testPath, index=False)

        print(f"Data split completed. Train: {len(train_data)}, Validation: {len(valid_data)}, Test: {len(test_data)}.")


    def readTrain(self) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, str]]]:
        """
        Reads and processes the training data from the specified path and returns long-term trajectory data
        and point-of-interest (POI) information.

        Returns:
        - longs (dict): A dictionary mapping user IDs to lists of long-term trajectories.
        - pois (dict): A dictionary mapping POI IDs to their latitude, longitude, and category.

        Raises:
        - FileNotFoundError: If the training data file does not exist.
        """
        if not os.path.exists(self.trainPath):
            raise FileNotFoundError(f"Training data file not found: {self.trainPath}")
        print(f"Trying to open training data file: {self.trainPath}")
        longs = dict()
        pois = dict()

        try:
            with open(self.trainPath, 'r') as file:
                lines = file.readlines()
            if self.check_data_empty(lines, "Training data"):
                raise ValueError("Training data is empty.")
            if self.datasetName == 'tky' and not self.check_data_format([line.strip().split(',') for line in lines[1:]], 9):
                raise ValueError("Training data format is incorrect.")
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
        except Exception as e:
            print(f"An error occurred while reading training data: {e}")
            raise
        return longs, pois

    def readTest(self) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, str]], Dict[str, Tuple[str, str]], Dict[str, str]]:

        """        
        Reads and processes the test data from the specified path and returns recent trajectory data,
        POI information, target data, and trajectory-to-user mapping.

        Returns:
        - recents (dict): A dictionary mapping trajectory IDs to recent trajectories.
        - pois (dict): A dictionary mapping POI IDs to their latitude, longitude, and category.
        - targets (dict): A dictionary mapping trajectory IDs to target POI.
        - traj2u (dict): A dictionary mapping trajectory IDs to user IDs.

        Raises:
        - FileNotFoundError: If the test data file does not exist.
        """
        if not os.path.exists(self.testPath):
            raise FileNotFoundError(f"Test data file not found: {self.testPath}")
        recents = dict()
        pois = dict()
        targets = dict()
        traj2u = dict()
        try:
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
        except Exception as e:
            print(f"An error occurred while reading test data: {e}")
            raise

        return recents, pois, targets, traj2u
    
    def getData(self) -> Dict[str, Any]:
        """
        Retrieves and processes both training and testing data, combining them into a single dataset.

        Returns:
        - data (dict): A dictionary containing:
            - "longs": Long-term trajectory data.
            - "recents": Recent trajectory data.
            - "targets": Target data.
            - "poiInfos": Combined POI information.
            - "traj2u": Trajectory to user mapping.
            - "poiList": List of unique POIs.

        Raises:
        - Exception: If there are errors during data reading.
        """
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

    def process(self) -> Dict[str, Any]:
        """
        Processes the data based on the dataset name and returns the processed data.

        Returns:
        - dict: Processed data based on the dataset.
        
        Raises:
        - ValueError: If the dataset name is not supported.
        """
        if self.datasetName in ['nyc', 'tky']:
            return self.getData()
        else:
            raise ValueError(f"Unsupported dataset: {self.datasetName}")
    
    def check_file_exists(self, file_path: str) -> bool:
        """
        Checks if the file exists at the given path.

        Parameters:
        - file_path (str): Path to the file.

        Returns:
        - bool: True if the file exists, False otherwise.
        """
        if os.path.exists(file_path):
            return True
        else:
            print(f"File does not exist: {file_path}")
            return False
        
    def check_data_empty(self, data: Any, data_name: str) -> bool:
        """
        Checks if the given data is empty.

        Parameters:
        - data (any): The data to check.
        - data_name (str): The name of the data being checked (for logging).

        Returns:
        - bool: True if the data is empty, False otherwise.
        """
        if not data:
            print(f"{data_name} is empty.")
            return True
        return False
    
    def check_data_format(self, data: List[List[str]], expected_columns: int) -> bool:
        """
        Checks if the data has the expected number of columns.

        Parameters:
        - data (list): List of data entries (rows).
        - expected_columns (int): Expected number of columns.

        Returns:
        - bool: True if the data has the expected number of columns, False otherwise.
        """
        for i, row in enumerate(data):
            if len(row) != expected_columns:
                print(f"Row {i+1} does not match the expected format. Expected {expected_columns} columns, found {len(row)}.")
                return False
        return True
    
    def check_user_exists(self, user_id: str, data: Dict[str, Any]) -> bool:
        """
        Checks if the given user ID exists in the data.

        Parameters:
        - user_id (str): The user ID to check.
        - data (dict): The dataset containing user IDs.

        Returns:
        - bool: True if the user ID exists, False otherwise.
        """
        if user_id in data:
            return True
        else:
            print(f"User ID {user_id} does not exist in the data.")
            return False







