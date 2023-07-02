from src.Student import StudentModel
from src.Teacher import TeacherModel
from src.utils import *
from src.args import parser
from src.incremental import knowledge_distillation, incremental_learning
from DATA.datasplit import split_tkg_by_timestamp
import time


# Press the green button in the gutter to run the script.
global_args = parser.parse_args()
if global_args.split:
    split_tkg_by_timestamp(load_data(global_args.filename))

def main():
    train_pair, dev_pair, adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features = load_data(global_args.filename, train_ratio=global_args.train_ratio)
    adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
    rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
    time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    time_size = time_features.shape[1]
    kg = set(adj_matrix, rel_matrix, ent_matrix, time_matrix, node_size, rel_size, time_size)
    teacher = TeacherModel(kg)
    student = StudentModel(kg)
    subgraphs = split_tkg_by_timestamp(kg)
    incremental_learning(teacher, student, subgraphs).to(global_args.device)
    for pairs in train_pair:
        tic = time.time()
        inputs = [adj_matrix, r_index, r_val, t_index, rel_matrix, ent_matrix]
        outputs = incremental_learning(inputs)
        loss = align_loss(pairs, output)
        loss.backward()
        toc = time.time()
    training_time = toc - tic
    return training_time



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
