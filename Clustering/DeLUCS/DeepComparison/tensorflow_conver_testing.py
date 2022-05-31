import tensorflow as tf
import pandas as pd
import numpy as np
def str_to_list(cell):
    sublist = []
    cell = ''.join(c for c in cell if c not in "'[]")
    
    cell = [float(item) for item in cell.split(',')]
    print(cell)
    return np.array(cell)

def load_dataset():
    datasets = pd.read_csv(r'C:\Users\User\Desktop\GithubCode\MLLib\Clustering\DeLUCS\DeepComparison\data\passes_sequence_coordinate.csv')

    for each_col in datasets.columns[1:]:
        datasets[each_col] = datasets[each_col].apply(str_to_list)
    datasets = datasets.iloc[1:,1:]
    print(datasets.values)
    return tf.convert_to_tensor(datasets.values.tolist())

dataset_processed = load_dataset()
print(dataset_processed[0])