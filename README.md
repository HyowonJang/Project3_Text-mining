## Project : Coupang review Text mining Project

## 1. Objective
쿠팡의 '외장하드' 100개 상품의 상품평을 크롤링하여 부정/긍정을 Classification하는 예측 모델을 만들고, 상품링크를 입력시 부정/긍정의 비율 및 핵심 키워드를 보여주는 프로젝트입니다.

## 2. Process
- Crawling : 쿠팡의 '외장하드' 상품평 크롤링
- Data Preprocessing : Konlpy Kkma & CountVectorize를 통해 유의미한 텍스트 데이터 전처리
- Modeling : Naive Bayes Multinomial
- Front-end : HTML/CSS/Javascipt를 활용하여 상품링크 입력 시 결과데이터 출력
- Back-end : Flask framework

## 3. Dataset Description
- train data : 쿠팡 '외장하드' 상위 100개 상품의 상품평 13806개
- test data 1 : [도시바 칸비오 어드밴스 외장하드 DTC920](http://www.coupang.com/vp/products/49867193)
- test data 2 : [WD Elements Portable 휴대용 외장하드 + 파우치](http://www.coupang.com/vp/products/25271683)


## 4. Modeling
- classification report - test data 1

NB Multi|precision|recall|f1-score|support
-----|--------|--------|-----|----
부정|0.67|1.00|0.80|2
긍정|1.00|0.99|1.00|128
avg/total|0.99|0.99|0.99|130

- classification report - test data 2

NB Multi|precision|recall|f1-score|support
-----|--------|--------|-----|----
부정|0.64|0.88|0.74|8
긍정|1.00|0.98|0.99|253
avg/total|0.98|0.98|0.98|261



## 5. Result
<img src="img/test data 1 result.png">

<img src="img/test data 1 result_1.png">
