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

# Read raw dataset
def read_datafile():
    data = pd.read_csv("data/WZUM_dataset.csv")
    
    return data

def visualize_data(data: pd.DataFrame):
    # Set list with colum names to get rid of unnecessary columns
    cols_x = ['world_landmark_' + str(i) + '.x' for i in range(21)]
    cols_y = ['world_landmark_' + str(i) + '.y' for i in range(21)]
    cols_z = ['world_landmark_' + str(i) + '.z' for i in range(21)]
    
    # hand_p_x = ['landmark_' + str(i*4 + 1) + '.x' for i in range(5)]
    # hand_p_y = ['landmark_' + str(i*4 + 1) + '.y' for i in range(5)]
    hand_p_z = ['landmark_' + str(i*4) + '.z' for i in range(0, 6)]
    cols_all_z = ['landmark_' + str(i) + '.z' for i in range(21)]
    for el in hand_p_z:
        cols_all_z.remove(el)
    
    fingertip_p_x = ['landmark_' + str(i*4 + 2) + '.x' for i in range(5)]
    fingertip_p_y = ['landmark_' + str(i*4 + 2) + '.y' for i in range(5)]
    fingertip_p_z = ['landmark_' + str(i*4 + 2) + '.z' for i in range(5)]
    # print(hand_p)
    
    # Set letters to numeric values along with labels
    data['letter'] = [letter_dict[letter] for letter in data['letter']]
    data['handedness.label'] = [hand_dict[label] for label in data['handedness.label']]
    
    # Drop not needed columns
    data.drop(['Unnamed: 0', 'handedness.score'], axis=1, inplace=True)
    data.drop(cols_x, axis=1, inplace=True)
    data.drop(cols_y, axis=1, inplace=True)
    data.drop(cols_z, axis=1, inplace=True)
    # data.drop(hand_p_x, axis=1, inplace=True)
    # data.drop(hand_p_y, axis=1, inplace=True)
    data.drop(cols_all_z, axis=1, inplace=True)
    data.drop(fingertip_p_x, axis=1, inplace=True)
    data.drop(fingertip_p_y, axis=1, inplace=True)
    # data.drop(fingertip_p_z, axis=1, inplace=True)

    # print effect of operations
    print(data)

    # Split to features and data
    y = data['letter']
    x = data.drop(['letter'], axis=1)

    return x, y

# Data saving function
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