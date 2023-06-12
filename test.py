import pandas as pd
import pickle
import argparse
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, f1_score


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
def read_datafile(datafile_path):
    data = pd.read_csv(datafile_path)
    
    return data

def prepare_data(data: pd.DataFrame):
    # Set list with colum names to get rid of unnecessary columns
    cols_x = ['world_landmark_' + str(i) + '.x' for i in range(21)]
    cols_y = ['world_landmark_' + str(i) + '.y' for i in range(21)]
    cols_z = ['world_landmark_' + str(i) + '.z' for i in range(21)]

    hand_p_z = ['landmark_' + str(i*4) + '.z' for i in range(0, 6)]
    cols_all_z = ['landmark_' + str(i) + '.z' for i in range(21)]
    for el in hand_p_z:
        cols_all_z.remove(el)
    
    fingertip_p_x = ['landmark_' + str(i*4 + 2) + '.x' for i in range(5)]
    fingertip_p_y = ['landmark_' + str(i*4 + 2) + '.y' for i in range(5)]
    
    # Set letters to numeric values along with labels
    data['letter'] = [letter_dict[letter] for letter in data['letter']]
    if ('handedness' in data.columns):
        data.drop(['Unnamed: 0', 'handedness'], axis=1, inplace=True)
    else: 
        data.drop(['Unnamed: 0', 'handedness.score', 'handedness.label'], axis=1, inplace=True)
    
    # Drop not needed columns
    data.drop(['Unnamed: 0', 'handedness.score'], axis=1, inplace=True)
    data.drop(cols_x, axis=1, inplace=True)
    data.drop(cols_y, axis=1, inplace=True)
    data.drop(cols_z, axis=1, inplace=True)
    data.drop(cols_all_z, axis=1, inplace=True)
    data.drop(fingertip_p_x, axis=1, inplace=True)
    data.drop(fingertip_p_y, axis=1, inplace=True)

    # Split to features and data
    y = data['letter']
    x = data.drop(['letter'], axis=1)

    return x, y

def save_results(y_predict, result_file_path):
    num_dict = dict()
    for key in letter_dict.keys():
        num_dict[letter_dict[key]] = key
    y_pred_labeled = [num_dict[val] for val in y_predict]
    df = pd.DataFrame({'letter':y_pred_labeled})
    df.to_csv(result_file_path, sep='\t', encoding='utf-8', index=False)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', type=str)
    parser.add_argument('result_file_path', type=str)
    args = parser.parse_args()
    data_file_path = Path(args.data_file_path)
    result_file_path = Path(args.result_file_path)
    model_path = os.path.realpath(os.path.dirname(__file__)) + "/models/model.pkl"
    data = read_datafile(data_file_path)
    x, y = prepare_data(data)
    x = x.to_numpy()

    model = pickle.load(open(model_path, 'rb'))
        
    y_predict = model.predict(x)
    
    save_results(y_predict, result_file_path)

    return 0

if __name__ == "__main__":
    main()