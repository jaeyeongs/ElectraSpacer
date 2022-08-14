from spaceprediction import ElectraSpacer
import pandas as pd

electraspacer = ElectraSpacer()

# Method 1
# input : wrong_sentence
# output : predict_sentence
# predict.py 실행을 하면 results 폴더에 predictions.json 으로 결과 저장
# 클래스를 직접 불러와 문장을 입력하여 곧 바로 띄어쓰기 예측
wrong_sentence = "안녕하세요저는연구소직원입니다.".replace(" ","")
correct_sentence = electraspacer.predict_sentence(wrong_sentence)

print("\n")
print("wrong :", wrong_sentence)
print("correct : ", correct_sentence)

# Method 2
# input : wrong_sentence
# output : predict_sentence
# predict.py 실행을 하면 results 폴더에 predictions.csv 으로 결과 저장
# 문장이 담긴 파일을 불러와서 띄어쓰기 교정을 한 결과를 다시 파일로 뽑아줌
# path = r'/workspace/ElectraSpacer/data/test1.csv'
# df = pd.read_csv(path, sep=',')
#
# space_sentence = []
#
# for sentence in df['wrong_sentence']:
#     sentence = electraspacer.predict_sentence(sentence)
#     space_sentence.append(sentence)
#
# complete_sentence = pd.DataFrame(space_sentence, columns=['predict_sentence'])
# complete_sentence = pd.concat([df,complete_sentence],axis=1)
# complete_sentence.to_csv('/workspace/ElectraSpacer/model/results/prediction.csv')



