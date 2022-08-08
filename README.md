# 한국어 자동 띄어쓰기 모델 : ElectraSpacer

## What is ElectraSpacer?

ElectraSpacer는 KoCharELECTRA를 사용하여 음절 단위로 tokenizer를 이용한 ELECTRA 모델 기반 한국어 자동 띄어쓰기 모델입니다.

## Dataset

국립국어원 [모두의 말뭉치](https://corpus.korean.go.kr "모두의 말뭉치")에서 문법성 판단 말뭉치를 학습 Dataset으로 사용했습니다.

**train_dataset.py**을 실행하여 json 파일을 csv 파일 형태로 바꿔 학습하였습니다.

- 문법성 판단 말뭉치
  - train.csv : 15,876개
  - dev.csv : 1,060개
  - test.csv : 1,060개
- 학습 데이터 구조
  - wrong_sentence : 띄어쓰기 공백이 모두 제거된 문장
  - correct_sentence : 올바른 띄어쓰기가 된 문장
  
## Model

| | KoBERT | KoCharELECTRA-Base | **KoCharELECTRA-Samll** |
|:---:|:---:|:---:|:---:|
| Accuracy | 97% | 98% | **99%** |
| F1 Score | 95% | 97% | **98%** |

자세한 내용은 **electra_architecture.pdf** 파일을 확인해주세요.

## Usage

### Tokenizer

ElectraSpacer는 아래와 같이 KoCharElectraTokenizer를 사용하여 character(음절) 단위로 토큰화합니다.

```
from tokenization_kocharelectra import KoCharElectraTokenizer
tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")
tokenizer.tokenize("나는 걸어가고 있는 중입니다.")

>> ['나', '는', ' ', '걸', '어', '가', '고', ' ', '있', '는', ' ', '중', '입', '니', '다', '.']
```

### Inference

**inference.py** 실행을 하면 results 폴더에 predictions.json 으로 결과가 저장됩니다.

```
"0": [
  "나는철수에게공을던져다주었다.",
  "나는 철수에게 공을 던져다 주었다.",
  "나는 철수에게 공을 던져다 주었다." 
]
```

### Predict

**predict.py** 실행을 하면 띄어쓰기 교정을 바로 확인 할 수 있습니다.

```
from spaceprediction import ElectraSpacer

electraspacer = ElectraSpacer()
electraspacer("나는걸어가고 있는중입니다.")

>> "나는 걸어가고 있는 중입니다."
```

## Reference

- [KoCharELECTRA](https://github.com/monologg/KoCharELECTRA "KoCharELECTRA")
- [ELECTRA](https://github.com/google-research/electra "ELECTRA")
- [Between-Spaces](https://github.com/boostcampaitech2/final-project-level3-nlp-03 "Between-Spaces")
