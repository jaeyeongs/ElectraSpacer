# 테스트 데이터 생성
# predict를 위한 틀린 문장(wrong_sentence)만 생성

import pandas as pd

path = r'/workspace/ElectraSpacer/data/train.csv'

df = pd.read_csv(path, sep=',')
df = df['wrong_sentence']

df.to_csv('/workspace/ElectraSpacer/data/test1.csv')

