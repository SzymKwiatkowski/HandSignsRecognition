import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

hand_dict = {
    'Right': 0,
    'Left': 1
}

letter_dict = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'k': 9,
    'l': 10,
    'm': 11,
    'n': 12,
    'o': 13,
    'p': 14,
    'q': 15,
    'r': 16,
    's': 17,
    't': 18,
    'u': 19,
    'v': 20,
    'w': 21,
    'x': 22,
    'y': 23,
}

def read_datafile():
    data = pd.read_csv("data/WZUM_dataset.csv")
    
    return data

def visualize_data(data: pd.DataFrame):
    data['letter'] = [letter_dict[letter] for letter in data['letter']]
    data['handedness.label'] = [hand_dict[label] for label in data['handedness.label']]
    data.drop(['Unnamed: 0', 
               'handedness.score', 
               'landmark_1.x', 'landmark_1.y', 'landmark_1.z',
               'landmark_3.x', 'landmark_3.y', 'landmark_3.z',
               'landmark_7.x', 'landmark_7.y', 'landmark_7.z',
               'landmark_11.x', 'landmark_11.y', 'landmark_11.z',
               'landmark_15.x', 'landmark_15.y', 'landmark_15.z',
               'landmark_19.x', 'landmark_19.y', 'landmark_19.z',
               'landmark_0.x', 'landmark_0.y', 'landmark_0.z',
               'handedness.label'
               ], 
              axis=1, inplace=True)
    print(data)

    y = data['letter']
    x = data.drop(['letter'], axis=1)

    return x, y

def save_data(x, y):
    f = open('data/data.pickle', 'wb')
    pickle.dump({'data': x, 'labels': y}, f)
    f.close()
    
    

def main():
    data = read_datafile()
    x, y = visualize_data(data)
    
    save_data(x, y)
    return 0

if __name__ == "__main__":
    main()