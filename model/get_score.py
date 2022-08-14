# split 된 test 데이터를 원 데이터와 비교하는 코드
# wer 같이 계산

import os
import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import jiwer

slot_labels = ["UNK", "PAD", "B", "I"]

def get_tags(sentence):
    all_tags = []
    for word in sentence:
        for i, c in enumerate(word):
            if i == 0:
                all_tags.append("B")
            else:
                all_tags.append("I")

    return all_tags

def getlabels(str_list):
    new_list = []
    for st in str_list:
        tags = get_tags(st.strip().split())
        tags = [slot_labels.index(t) for t in tags]
        new_list.append(tags)
    return new_list

def compute_real_metric(test_data_path, prediction_path):
    df = pd.read_csv(test_data_path)
    with open(prediction_path, encoding='utf-8') as f:
        pred_dict = json.load(f)
    
    print(len(df), len(pred_dict))
    
    join_sentences = join_sentence(df, pred_dict)
    
    assert len(df)==len(join_sentences)
    
    df['predicted'] = join_sentences
    
    df['pred_labels'] = getlabels(df['predicted'])
    df['correct_labels'] = getlabels(df['correct_sentence'])
    
    metrics = compute_metrics(df)
    
    return metrics
    
def join_sentence(df, pred_dict):
    total_list = []
    temp_list = []

    df_idx = 0
    for key in pred_dict:
        wrong = pred_dict[key][0]
        correct = pred_dict[key][1]
        pred = pred_dict[key][2]

        if correct in df.loc[df_idx]['correct_sentence']:
            temp_list.append(pred)

        else:
            total_list.append(", ".join(temp_list))
            temp_list = []
            df_idx += 1
            temp_list.append(pred)

    total_list.append(", ".join(temp_list))

    return total_list

def compute_metrics(df):
    accuracy = 0
    accuracy_aon = 0
    f1 = 0
    pred_len = len(df)
    cnt = 0
    WER = 0

    for idx, row in df.iterrows():
        pred_list = row['pred_labels']
        label_list = row['correct_labels']
    
        try:
            accuracy += accuracy_score(label_list, pred_list)
            accuracy_aon += (label_list == pred_list)
            f1 += f1_score(pred_list, label_list, pos_label=2)
            WER += jiwer.wer(row['correct_sentence'], row['predicted'])
        except:
            cnt+=1
            
    return {'acc': accuracy/pred_len, 
            'acc_binary': accuracy_aon/pred_len, 
            'f1_score': f1/pred_len,
            'cnt': cnt,
            'WER': WER/pred_len  # average of WER
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data", default='../data/reduced_test_v3.csv')
    parser.add_argument("--pred_data", default='./results/predictions.json')
    args = parser.parse_args()

    print(compute_real_metric(args.test_data, args.pred_data))
