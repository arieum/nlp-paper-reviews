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
### 모델이 다음 단어를 하나씩 생성할 때, 어떤 기준으로 단어를 고를지 정하는 방법론
1. **그리디 서치**: 매 시점에서 가장 확률이 높은 단어 하나만 선택하는 방법 → ```do_sample=False, num_beams=1```
   ```python
    import torch
   
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False, # 확률값이 높은 단어를 다음 단어로 결정
            min_length=10,
            max_length=50,
        )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
    ```
3. **빔 서치**: 매 단계에서 확률 상위 k개(beam size)의 후보 문장을 유지 → ```do_sample=False, num_beams=3```
    ```python
    with torch.no_grad():
      generated_ids = model.generate(
          input_ids,
          do_sample=False,
          min_length=10,
          max_length=50,
          num_beams=3,  # 빔 크기=3 지정
      )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
    ```
### 반복되는 표현 줄이기
언어모델로 문장을 생성할 때 자주 아래와 같은 현상 발생

∵ GPT류 모델이 이전에 잘 나온 단어를 계속 이어 쓰는 경향때문(확률 분포 상 같은 패턴이 자꾸 높은 점수를 받기 때문)

```
“그건 뭐예요? 그건 뭐예요? 그건 뭐예요?...”
“좋아요 좋아요 좋아요...”
```

sol: ```no_repeat_ngram_size```(n개의 연속된 토큰이 다시 등장하지 않게 하는 규칙)
  _ex. no_repeat_ngram_size=3 → 3-gram 단위로 검사_
  
<details>
<summary>예시</summary>
모델이 문장을 이렇게 만들고 있다면:
  
```arduino
입력: "그럼"
생성 중: "그럼, 그건 뭐예요?"
```

이제 다음 단어를 예측하려 할 때,
현재 생성된 문장 안에 “그건 뭐예요”라는 3-gram(3단어 묶음)이 이미 존재함

그래서 만약 모델이 또 “그건 뭐예요”를 이어 붙이려 하면,
→ 그 n-gram 확률을 0으로 만들어버려서 선택할 수 없게함

결과적으로 “그건 뭐예요?” 같은 반복 문장이 안 나오게 되는 원리
</details>

### Top-k 샘플링
> 다음 단어를 뽑을 때 **확률값 기준 가장 큰 k개 가운데 하나를 선택하는 기법** 

확률값이 큰 단어가 다음 단어로 뽑힐 가능성이 높아지지만, k개 안에 있는 단어라면 확률값이 낮더라도 다음 단어로 추출될 수 있음. 
따라서 top-k sampling은 매 시행 때마다 생성 결과가 달라짐.

```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_k=50,
    )
print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

### Temperature Scaling
> **모델의 다음 토큰 확률 분포를 대소 관계의 역전없이 분포의 모양만 바꿔서 문장을 다양하게 생성하는 기법**

모델의 출력 로짓(소프트맥스 변환 전 벡터)의 모든 요솟값을 temperature로 나누는 방식

ex. temperature=2, [-1.0, 2.0, 3.0] -> [-0.5, 1.0, 1.5]
▶︎ 0에 가까울 수록 확률분포모양이 원래보다 뾰족해짐(원래 컸던 확률은 더 커지고, 작았던 확률은 더 작아짐) → 정확한 문장
▶︎ 1보다 클수록 확률분포모양이 평평해짐(원래 컸던 확률과 작았던 확률 사이의 차이가 줄어듦) → 다양한 문장

※ temperature=1이면 변화 X

※ temperature 음수면 대소관계가 바뀜

※ temperature scailing은 top-p, top-k와 함께 적용해야 의미가 있음


```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_k=50,
        temperature=0.01,
    )
print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

### Top-p Sampling ≅ nucleus sampling
> **확률값이 높은 순서대로 내림차순 정렬 한 뒤, 누적 확률값이 p 이상인 최소 갯수의 토큰 집합 가운데 하나를 다음 토큰으로 선태하는 기법**


```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_p=0.92,
    )
print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```
