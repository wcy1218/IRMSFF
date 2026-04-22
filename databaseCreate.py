import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

MODEL_NAME = "model/unixcoder" # 本地模型文件

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

N_COMPONENTS = 256
MAX_LENGTH = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(name):
    tokenizer = RobertaTokenizer.from_pretrained(name)
    model = RobertaModel.from_pretrained(name)
    model = model.to(DEVICE)
    return tokenizer, model

#向量化
def sents_to_vecs(sents, tokenizer, model):
    vecs = []
    with torch.no_grad():
        for sent in sents:
        # for sent in tqdm(sents):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = hidden_states[-1][0] + hidden_states[1][0]
            elif POOLING == 'last_avg':
                output_hidden_state = hidden_states[1][0]
            elif POOLING == 'last2avg':
                output_hidden_state = hidden_states[-1][0] + hidden_states[-2][0]
            else:
                raise Exception("unknown pooling {}".format(POOLING))
            # output_hidden_state [batch_size, hidden_size]
            vec = output_hidden_state.cpu().numpy()
            vecs.append(vec)
    assert len(sents) == len(vecs)
    print(len(vecs))
    # vecs = np.array(vecs)
    return vecs

#对矩阵的第二维进行归一化，使每个词向量长度为1
def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def main():
    print(f"Configs: {MODEL_NAME}-{POOLING}--{N_COMPONENTS}.")
    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

    #选择数据集
    df = pd.read_csv("data/train_code.csv", header=None)

    #选择数量
    code_list = df[0].tolist()
    print("Transfer sentences to BERT vectors.")
    vecs_func_body = sents_to_vecs(code_list, tokenizer, model) # [code_list_size, code_size, 768]
    # print(vecs_func_body[0].shape)
    for i in range(len(vecs_func_body)):
        vecs_func_body[i] = normalize(vecs_func_body[i]) # [code_size, 768]
    import pickle
    f = open('model/code_vector_test.pkl', 'wb')
    pickle.dump(vecs_func_body, f)
    f.close()


if __name__ == "__main__":
    main()
