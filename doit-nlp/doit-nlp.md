- 💡 [Tokenizer](#tokenizer)
  - [VER.GPT_Tokenizer](#vergpt_tokenizer)
    - [1. Vocab 구축하기](#1-vocab-구축하기)
    - [2. Vocab 토대로 입력값 토큰화](#2-vocab-토대로-입력값-토큰화)
  - [VER.BERT_Tokenizer](#verbert_tokenizer)
    - [1. Vocab 구축하기](#1-vocab-구축하기-1)
    - [2. Vocab 토대로 입력값 토큰화](#2-vocab-토대로-입력값-토큰화-1)
- 💡 [Embedding(Representation)](#embeddingrepresentation)
  - [1. Tokenizer 선언](#1-tokenizer-선언)
  - [2. BERT 모델 초기화](#2-bert-모델-초기화)
  - [3. 입력값 초기화](#3-입력값-초기화)
  - [4. BERT 기반 단어/문장 수준 벡터](#4-bert-기반-단어문장-수준-벡터)
- 💡 [Sentence Generation](#sentence-generation)

# Tokenizer
## VER.GPT_Tokenizer
### 1. Vocab 구축하기
```python
from tokenizers import ByteLevelBPETokenizer

bytebpe_tokenizer = ByteLevelBPETokenizer()

bytebpe_tokenizer.train(
    files=["/content/train.txt", "/content/test.txt"],
    vocab_size=10000,
    special_tokens=["[PAD]"]
)

bytebpe_tokenizer.save_model("/gdrive/My Drive/nlpbook/bbpe")
```
결과
- ```vocab.json```: Byte 수준 BPE 기반 vocab
  - 문자 단위가 아닌 유니코드 바이트 수준으로 어휘집합을 구축하고 토큰화 수행 → 미등록 토큰 문제 완화 
- ```merges.txt```: Bigram 쌍의 병합우선순위

### 2. Vocab 토대로 입력값 토큰화
```python
from transformers import GPT2Tokenizer
tokenizer_gpt = GPT2Tokenizer.from_pretrained("/gdrive/My Drive/nlpbook/bbpe")
tokenizer_gpt.pad_token = "[PAD]"

sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]
batch_inputs = tokenizer_gpt(
    sentences,
    padding="max_length",  # 문장 최대 길에 맞춰 패딩
    max_length=12,         # 문장의 토큰 기준 최대 길이
    truncation=True,       # 문장 잘림 허용 옵션
)
```
결과
- ```batch_inputs['input_ids']```: input_ids는 각 문장마다 토큰화 결과를 가지고 각 토큰을 인덱스로 바꾼 것
- ```batch_inputs['attention_mask']```: attention_mask는 일반토큰이 자리한 곳(1)과 패딩 토큰이 자리한 곳(0)을 알려주는 장치

## VER.BERT_Tokenizer
### 1. Vocab 구축하기
```python
from tokenizers import BertWordPieceTokenizer

wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)

wordpiece_tokenizer.train(
    files=["/content/train.txt", "/content/test.txt"],
    vocab_size=10000,
)

wordpiece_tokenizer.save_model("/gdrive/My Drive/nlpbook/wordpiece")
```
결과
- ```vocab.txt```: WordPiece 기반 vocab

### 2. Vocab 토대로 입력값 토큰화
```python
from transformers import BertTokenizer
tokenizer_bert = BertTokenizer.from_pretrained(
    "/gdrive/My Drive/nlpbook/wordpiece", 
    do_lower_case=False,
)

sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]
batch_inputs = tokenizer_gpt(
    sentences,
    padding="max_length",  # 문장 최대 길에 맞춰 패딩
    max_length=12,         # 문장의 토큰 기준 최대 길이
    truncation=True,       # 문장 잘림 허용 옵션
)
```
결과
- ```batch_inputs['input_ids']```: input_ids는 각 문장마다 토큰화 결과를 가지고 각 토큰을 인덱스로 바꾼 것
  - BERT는 문장 시작과 끝에 [CLS](2), [SEP](3) 토큰을 덧붙임 
- ```batch_inputs['attention_mask']```: attention_mask는 일반토큰이 자리한 곳(1)과 패딩 토큰이 자리한 곳(0)을 알려주는 장치
- ```batch_inputs['token_type_ids']```: 세그먼트 정보 (한 배열[]내에서 첫번째 세그먼트(문서, 문장)는 0, 두번째 세그먼트는 1, ...)

# Embedding(Representation)
### 1. Tokenizer 선언
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "beomi/kcbert-base",
    do_lower_case=False,
)
```
### 2. BERT 모델 초기화
```python
from transformers import BertConfig, BertModel

pretrained_model_config = BertConfig.from_pretrained(
    "beomi/kcbert-base"
)
model = BertModel.from_pretrained(
    "beomi/kcbert-base",
    config=pretrained_model_config,
)
```
### 3. 입력값 초기화
```python
sentences = ["안녕하세요", "하이!"]

features = tokenizer(
    sentences,
    max_length=10,
    padding="max_length",
    truncation=True,
)
```
결과
- ```features['input_ids']```: 2개의 입력문장 각각에 대해 WordPiece 토큰화를 수행한 뒤, 이를 토큰 인덱스로 변환한 결과
  
    ```
    [[2, 19017, 8482, 3, 0, 0, 0, 0, 0, 0], # "안녕하세요" 
    [2, 15830, 5, 3, 0, 0, 0, 0, 0, 0]]     # "하이!"
    ```
  
- ```features['attention_mask']```: 패딩이 아닌 토큰은 1, 패딩인 토큰은 0으로 / 실제 토큰이 자리하는지 아닌지의 정보
   
    ```
    [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    ```
### 4. BERT 기반 단어/문장 수준 벡터
```python
# Pytorch의 입력값은 Tensor이어야 함
import torch
features = {k: torch.tensor(v) for k, v in features.items()}

outputs = model(**features)

outputs.last_hidden_state
```
➮ **문장 2개**에 속한 **각각의 토큰(길이 10)**이 **768차원의 벡터**로 변환됨

# Sentence Generation
