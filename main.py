import pandas as pd
import json

NUM_OF_REDUNDANT_HEADER_ROWS = 19
MS_COLUMN = 'Timestamp'
INTEROCULAR_DISTANCE_COLUMN= 'Interocular Distance'
CONFIG_PATH = 'config.json'
TIME_UNIT_TO_MEASURE = 1000 # one_second

def read_imotion_file_input_as_data_frame(file_name, score_col_name):
    data_frame = pd.read_csv(file_name, header=1, index_col=MS_COLUMN, skiprows=NUM_OF_REDUNDANT_HEADER_ROWS, usecols=[MS_COLUMN, INTEROCULAR_DISTANCE_COLUMN, score_col_name])
    return data_frame

def calculate_score_vector(data_frame, score_col_name, score_threshold):
    score_vector = data_frame.groupby(lambda time: time // TIME_UNIT_TO_MEASURE)[score_col_name].apply(lambda score_column: (score_column > score_threshold).sum() / len(score_column))
    return score_vector

def align_vectors(first, second):
    shortest_max_idx = min(first.index.max(), second.index.max())
    return [ s.reindex(range(shortest_max_idx + 1)) for s in [first, second] ]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_frames = []
    vectors = []
    groups = []
    conf = open(CONFIG_PATH)
    conf = json.load(conf)
    for file in [conf['target_file'], conf['regulator_file']]:
        data_frame = read_imotion_file_input_as_data_frame(file, score_col_name=conf['score_column'])
        data_frame = data_frame[data_frame[INTEROCULAR_DISTANCE_COLUMN] != 0] # filter invalid rows

        vector = calculate_score_vector(data_frame,
                                        score_col_name=conf['score_column'],
                                        score_threshold=conf['score_threshold'])
        data_frames.append(data_frame)
        vectors.append(vector)

    target_vector, regulator_vector = align_vectors(*vectors)
    correlation = target_vector.corr(regulator_vector)
    print(correlation)

