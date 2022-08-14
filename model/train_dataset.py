# 학습 데이터 생성
# 올바른 문장(correct_sentence)에서 공백을 모두 제거한 틀린 문장(wrong_sentence) 생성

import pandas as pd

path = r'/ElectraSpacer\data\dw-train.csv'

df = pd.read_csv(path, sep=',')
df = pd.DataFrame(df['sentence'])
df = df.rename(columns={'sentence':'correct_sentence'})

nospace_sentence = []

for sentence in df['correct_sentence']:
    sentence = sentence.replace(" ","")
    nospace_sentence.append(sentence)

nospace = pd.DataFrame(nospace_sentence, columns=['wrong_sentence'])

complete_sentence = pd.concat([df,nospace],axis=1)
complete_sentence.to_csv('/workspace/ElectraSpacer/data/dw-train.csv')

