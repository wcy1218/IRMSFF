import pickle

import torch
import numpy as np

from nlgeval import compute_metrics
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from databaseCreate import sents_to_vecs, normalize

import heapq
import os
# import bert_score
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#token最大长度
dim = 256
POOLING = 'first_last_avg'

df = pd.read_csv("data/train_code.csv", header=None)
train_code_list = df[0].tolist()

df = pd.read_csv("data/train_nl.csv", header=None)
train_nl_list = df[0].tolist()

df = pd.read_csv("data/test_code.csv", header=None)
test_code_list = df[0].tolist()

df = pd.read_csv("data/test_nl.csv", header=None)
test_nl_list = df[0].tolist()

tokenizer = RobertaTokenizer.from_pretrained("model/unixcoder")
model = RobertaModel.from_pretrained("model/unixcoder")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

def bert_score_plus(code1, code2):
    code1 = torch.from_numpy(code1)
    code2 = torch.from_numpy(code2)
    code2 = torch.transpose(code2, 0, 1)
    matrix = torch.mm(code1, code2)
    sum = 0
    for i in range(1,matrix.size()[0]-1):
        first_max = -2
        second_max = -2
        for j in range(1,matrix.size()[1]-1):
            if matrix[i][j] > first_max:
                second_max = first_max
                first_max = matrix[i][j]
            elif matrix[i][j] > second_max:
                second_max = matrix[i][j]
        sum += first_max*2 - second_max

    for i in range(1,matrix.size()[1]-1):
        first_max = -2
        second_max = -2
        for j in range(1,matrix.size()[0]-1):
            if matrix[j][i] > first_max:
                second_max = first_max
                first_max = matrix[j][i]
            elif matrix[j][i] > second_max:
                second_max = matrix[j][i]
        sum += first_max*2 - second_max
    score = sum / (matrix.size()[0]+matrix.size()[1])
    return score

def bert(code1,code2):
    code1 = torch.from_numpy(code1).mean(dim=0)
    norm = torch.norm(code1)
    code1 = code1 / norm
    code2 = torch.from_numpy(code2).mean(dim=0)
    norm = torch.norm(code2)
    code2 = code2 / norm
    return torch.dot(code1,code2)


def bert_score_my(code1, code2):
    code1 = torch.from_numpy(code1)
    code2 = torch.from_numpy(code2)
    code2 = torch.transpose(code2, 0, 1)
    matrix = torch.mm(code1, code2)
    sum = 0
    for i in range(1,matrix.size()[0]-1):
        max = -2
        for j in range(1,matrix.size()[1]-1):
            if matrix[i][j] > max:
                max = matrix[i][j]
        sum += max
    for i in range(1,matrix.size()[1]-1):
        max = -2
        for j in range(1,matrix.size()[0]-1):
            if matrix[j][i] > max:
                max = matrix[j][i]
        sum += max
    score = sum / (matrix.size()[0]+matrix.size()[1])
    return score

def bert_score_my1(code1, code2):
    code1 = torch.from_numpy(code1)
    code2 = torch.from_numpy(code2)
    code2 = torch.transpose(code2, 0, 1)
    matrix = torch.mm(code1, code2)
    sum = 0
    for i in range(1,matrix.size()[0]-1):
        max = -2
        for j in range(1,matrix.size()[1]-1):
            if matrix[i][j] > max:
                max = matrix[i][j]
        sum += max
    score1 = sum / (matrix.size()[0]-2)

    sum = 0
    for i in range(1,matrix.size()[1]-1):
        max = -2
        for j in range(1,matrix.size()[0]-1):
            if matrix[j][i] > max:
                max = matrix[j][i]
        sum += max
    score2 = sum / (matrix.size()[1]-2)
    score = (2*score1*score2)/(score1+score2)
    return score

def bert_score_idf(code1, code2, idx_text1, idx_text2):
    code1 = torch.from_numpy(code1)
    code2 = torch.from_numpy(code2)
    code2 = torch.transpose(code2, 0, 1)
    matrix = torch.mm(code1, code2)
    sum = 0
    for i in range(1,matrix.size()[0]-1):
        max = -2
        for j in range(1,matrix.size()[1]-1):
            if matrix[i][j] > max:
                max = matrix[i][j]
        sum += max
    for i in range(1,matrix.size()[1]-1):
        max = -2
        for j in range(1,matrix.size()[0]-1):
            if matrix[j][i] > max:
                max = matrix[j][i]
        sum += max
    score = sum / (matrix.size()[0]+matrix.size()[1])
    return score

def largest_k_elements_with_indices(list, k):
    # 使用heapq.nlargest找到最大的k个值
    largest_k_values = heapq.nlargest(k, list)
    # 找到这些值的下标
    indices = [i for i, value in enumerate(list) if value in largest_k_values]
    return indices

def main():
    f = open('model/code_vector_my.pkl', 'rb')
    bert_vec = pickle.load(f)
    f.close()
    all_texts = []
    all_ids = []
    all_vecs = []
    for i in range(1000):
    # for i in range(len(bert_vec)):
        all_texts.append(train_code_list[i])
        all_ids.append(i)
        all_vecs.append(bert_vec[i])
    id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
    ids = np.array(all_ids, dtype="int64")

    sim_nl_list = []
    nl_list = []
    for i in tqdm(range(150)):
        scores = []
        for j in range(len(bert_vec)):
            code_score = sim_jaccard(train_code_list[j].split(),test_code_list[i].split())
            scores.append(code_score)
        idx = largest_k_elements_with_indices(scores,20)

        test_vec = sents_to_vecs([test_code_list[i]], tokenizer, model)

        #####去除首尾token

        bert_score_max = 0
        max_idx = 0
        for k in range(len(idx)):
            if bert_score_my1(bert_vec[idx[k]],test_vec[0]) > bert_score_max:
                bert_score_max = bert_score_my1(bert_vec[idx[k]], test_vec[0])
                max_idx = idx[k]
        sim_nl_list.append(train_nl_list[max_idx])
        nl_list.append(test_nl_list[i])

    df = pd.DataFrame(nl_list)
    df.to_csv("nl_my.csv", index=False, header=None)
    df = pd.DataFrame(sim_nl_list)
    df.to_csv("sim_my.csv", index=False, header=None)

    metrics_dict = compute_metrics(hypothesis='sim_my.csv',
                                   references=['nl_my.csv'], no_skipthoughts=True, no_glove=True)

if __name__ == "__main__":
    main()
