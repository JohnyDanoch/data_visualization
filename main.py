import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

NUM_OF_REDUNDANT_HEADER_ROWS = 19
MS_COLUMN = 'Timestamp'
INTEROCULAR_DISTANCE_COLUMN = 'Interocular Distance'
ATTENTION_COLUMN = 'Attention'
CONFIG_PATH = 'config.json'
TIME_UNIT_TO_MEASURE = 300  # one_second
TIME_UNIT_TO_ROLL= 300  # one_second

def read_imotion_file_input_as_data_frame(file_name, score_col_name):
    data_frame = pd.read_csv(file_name, header=1, index_col=MS_COLUMN, skiprows=NUM_OF_REDUNDANT_HEADER_ROWS, usecols=[MS_COLUMN, INTEROCULAR_DISTANCE_COLUMN, ATTENTION_COLUMN, score_col_name])
    return data_frame

def calculate_score_vector(data_frame, score_col_name, score_threshold):
    score_vector = data_frame.groupby(lambda time: time // TIME_UNIT_TO_MEASURE)[score_col_name].apply(lambda score_column: (score_column > score_threshold).sum() / len(score_column))
    return score_vector

def calculate_score_vector_mean(data_frame, score_col_name):
    score_vector = data_frame.groupby(lambda time: time // TIME_UNIT_TO_MEASURE)[score_col_name].mean()
    return score_vector

def roll_score_column(data_frame, score_col_name):
    data_frame[score_col_name] = data_frame[score_col_name].rolling(TIME_UNIT_TO_ROLL, min_periods=3).mean()


def align_vectors(first, second):
    shortest_max_idx = min(first.index.max(), second.index.max())
    return [s.reindex(range(shortest_max_idx + 1)) for s in (first, second)]

def filter_missing_data(data_frame):
    filtered_frame = data_frame[data_frame[INTEROCULAR_DISTANCE_COLUMN] != 0]
    filtered_frame = filtered_frame[filtered_frame[ATTENTION_COLUMN] >= 35]

    return filtered_frame

def create_pdf_given_score_vectors(couple_number, target, regulator, correlation, score_col_name):
    plt.clf()
    plt.figure(1, figsize=(12, 10))
    plt.suptitle(f'Pearson Correlation: {correlation}', fontsize=14, fontweight='bold')
    plt.scatter(target.index, target.values, color='green', label='target', marker='|', s=15)
    plt.scatter(regulator.index, regulator.values, label='regulator', marker='.', s=7)
    plt.ylabel(f'{score_col_name} occurrence')
    plt.xlabel('300 milli seconds')
    plt.legend()
    plt.xticks(np.arange(0, len(target) + 1, 100))
    plt.savefig(f"./results/{couple_number}_graph.pdf")

def create_pdf_given_input_dataframe(input_file_name, target, regulator, score_col_name):
    plt.clf()
    plt.figure(1, figsize=(12, 4))
    plt.scatter(target.index, target[score_col_name], color='green', label='target', marker='|', s=15)
    plt.scatter(regulator.index, regulator[score_col_name], label='regulator', marker='.', s=7)
    plt.ylabel(f'{score_col_name} score')
    plt.xlabel('milli seconds')
    plt.legend()
    plt.savefig(f"./results/{input_file_name}")

def read_data_frame(file_name, score_col_name):
    data_frame = read_imotion_file_input_as_data_frame(file_name, score_col_name)
    data_frame = filter_missing_data(data_frame)
    return data_frame

if __name__ == '__main__':
    conf = open(CONFIG_PATH)
    conf = json.load(conf)
    data_path: str = conf['data_dir']
    couples_number_to_file_names = defaultdict(dict)

    for file_name in os.listdir(data_path):
        if not file_name.endswith('.csv'):
            raise ValueError(f'Only csv files are supported, got {file_name}')
        couple_number = file_name.split('_')[1]
        is_target = file_name.split('_')[2] == 'Target'
        couples_number_to_file_names[couple_number]['target' if is_target else 'regulator'] = file_name

    # validate all couples contains both target and regulator
    for couple_number, file_names in couples_number_to_file_names.items():
        if 'target' not in file_names or 'regulator' not in file_names:
            raise ValueError(f'Couple {couple_number} is missing target or regulator')

    for couple_number, files in couples_number_to_file_names.items():
    # for couple_number, files in [('13',  couples_number_to_file_names['13'])]:
        data_frames = {'target': read_data_frame('/'.join(['.', data_path, files['target']]), conf['score_column']),
                       'regulator': read_data_frame('/'.join(['.', data_path, files['regulator']]), conf['score_column'])}
        input_file_name = f'{couple_number}_raw_data.pdf'
        create_pdf_given_input_dataframe(input_file_name, data_frames['target'], data_frames['regulator'], conf['score_column'])
        roll_score_column(data_frames['target'], conf['score_column'])
        roll_score_column(data_frames['regulator'], conf['score_column'])
        input_file_name = f'{couple_number}_after_roll.pdf'
        create_pdf_given_input_dataframe(input_file_name, data_frames['target'], data_frames['regulator'], conf['score_column'])
        vectors = {'target': calculate_score_vector_mean(data_frames['target'], conf['score_column']),
                   'regulator': calculate_score_vector_mean(data_frames['regulator'], conf['score_column'])}
        vectors = align_vectors(vectors['target'], vectors['regulator'])
        correlation = vectors[0].corr(vectors[1])
        create_pdf_given_score_vectors(couple_number, vectors[0], vectors[1], correlation, conf['score_column'])
        print(f"Couple: {couple_number}, Correlation: {correlation}, Target number of samples: {len(data_frames['target'])}, Regulator number of samples: {len(data_frames['regulator'])}")
