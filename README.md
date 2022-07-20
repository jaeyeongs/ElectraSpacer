# 한국어 자동 띄어쓰기 모델 : ElectraSpacer

## What is ElectraSpacer?

ElectraSpacer는 KoCharELECTRA를 사용하여 Wordpiece-level이 아닌 Character-level(음절) tokenizer를 이용하여 학습한 ELECTRA 모델 기반 한국어 자동 띄어쓰기 모델입니다.

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
