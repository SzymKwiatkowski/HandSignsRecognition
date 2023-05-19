import pandas as pd
import matplotlib
import seaborn as sb
import numpy as np

def read_datafile():
    data = pd.read_csv("data/WZUM_dataset.csv")
    
    return data

def visualize_data(data: pd.DataFrame):
    print(data.describe())
    print(data)
    
    

def main():
    data = read_datafile()
    visualize_data(data)
    
    return 0

if __name__ == "__main__":
    main()