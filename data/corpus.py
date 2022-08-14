# 말뭉치 Dataset를 다운 받을 수 있는 Korpora 라이브러리
# pip install Korpora
# git clone https://github.com/ko-nlp/Korpora
# python setup.py install

from Korpora import Korpora

# print(Korpora.fetch('all'))

corpus = Korpora.load("namuwikitext")
a = corpus.get_all_texts()

for line in a:
    print(line)

# with open("/root/Korpora/kcbert/20190101_20200611_v2.txt", 'r', encoding="utf-8") as file:
#
#     while True:
#         print(file.readlines())

# 말뭉치 List
# {
#    'kcbert': 'beomi@github 님이 만드신 KcBERT 학습데이터',
#    'korean_chatbot_data': 'songys@github 님이 만드신 챗봇 문답 데이터',
#    'korean_hate_speech': '{inmoonlight,warnikchow,beomi}@github 님이 만드신 혐오댓글데이터',
#    'korean_petitions': 'lovit@github 님이 만드신 2017.08 ~ 2019.03 청와대 청원데이터',
#    'kornli': 'KakaoBrain 에서 제공하는 Natural Language Inference (NLI) 데이터',
#    'korsts': 'KakaoBrain 에서 제공하는 Semantic Textual Similarity (STS) 데이터',
#    'kowikitext': "lovit@github 님이 만드신 wikitext 형식의 한국어 위키피디아 데이터",
#    'namuwikitext': 'lovit@github 님이 만드신 wikitext 형식의 나무위키 데이터',
#    'naver_changwon_ner': '네이버 + 창원대 NER shared task data',
#    'nsmc': 'e9t@github 님이 만드신 Naver sentiment movie corpus v1.0',
#    'question_pair': 'songys@github 님이 만드신 질문쌍(Paired Question v.2)',
#    'modu_news': '국립국어원에서 만든 모두의 말뭉치: 뉴스 말뭉치',
#    'modu_messenger': '국립국어원에서 만든 모두의 말뭉치: 메신저 말뭉치',
#    'modu_mp': '국립국어원에서 만든 모두의 말뭉치: 형태 분석 말뭉치',
#    'modu_ne': '국립국어원에서 만든 모두의 말뭉치: 개체명 분석 말뭉치',
#    'modu_spoken': '국립국어원에서 만든 모두의 말뭉치: 구어 말뭉치',
#    'modu_web': '국립국어원에서 만든 모두의 말뭉치: 웹 말뭉치',
#    'modu_written': '국립국어원에서 만든 모두의 말뭉치: 문어 말뭉치',
#    'aihub_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (구어 + 대화 + 뉴스 + 한국문화 + 조례 + 지자체웹사이트)",
#    'aihub_spoken_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (구어)",
#    'aihub_conversation_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (대화)",
#    'aihub_news_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (뉴스)",
#    'aihub_korean_culture_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (한국문화)",
#    'aihub_decree_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (조례)",
#    'aihub_government_website_translation': "AI Hub 에서 제공하는 번역용 병렬 말뭉치 (지자체웹사이트)",
#    'open_subtitles': 'Open parallel corpus (OPUS) 에서 제공하는 영화 자막 번역 병렬 말뭉치',
# }

