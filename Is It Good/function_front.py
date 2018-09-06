
import pandas as pd
import numpy as np
import requests
from scrapy.http import TextResponse
import json
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Kkma
import pickle

def review_cnt_check(keyword, item_count_start=0, item_count_end=100):

    search_url = "http://www.coupang.com/np/search?q={}&isPriceRange=false&page=1&sorter=scoreDesc&listSize=100".format(keyword)
    rep = requests.get(search_url)
    response = TextResponse(rep.url, body=rep.text, encoding='utf-8')

    # 상품마다의 링크를 불러오기에 앞서 아이템 Top100 리스트 불러오기
    products = json.loads(response.xpath('//*[@id="productList"]/@data-products').extract()[0])['indexes'][item_count_start:item_count_end]

    review_total_cnt = 0

    for idx, product in enumerate(products):

        # Top100 상품마다의 상품평 개수(나중에 불러오는 정보라 상품링크 안에서는 xpath로 바로 가져올 수 없음)
        try:
            review_count = int(response.xpath('//*[@id="{}"]/a/dl/dd/div/div[4]/div[2]/span[2]/text()'.format(product)).extract()[0].strip("()"))
            review_total_cnt += review_count
            print('{}번째 item {}의 review 수 : {}'.format(idx, product, review_count))

        except Exception as e:
            print('No reviews')

    print('{}번째부터 {}번째 item의 총 review 수 : {}'.format(item_count_start, item_count_end-1, review_total_cnt))

def make_df_train(keyword, item_count_start=0, item_count_end=100):

    search_url = "http://www.coupang.com/np/search?q={}&isPriceRange=false&page=1&sorter=scoreDesc&listSize=100".format(keyword)
    rep = requests.get(search_url)
    response = TextResponse(rep.url, body=rep.text, encoding='utf-8')

    # 상품마다의 링크를 불러오기에 앞서 아이템 Top100 리스트 불러오기
    products = json.loads(response.xpath('//*[@id="productList"]/@data-products').extract()[0])['indexes'][item_count_start:item_count_end]
    products = list(set(products))
    print("총 {}개의 상품에서 중복제거 후 {}개의 상품 크롤링".format(item_count_end-item_count_start,len(products)))

    df = pd.DataFrame(columns=['rating', "item_nbr", "item", "option", "date", "name", "re_title", 'review'])

    top100_review_total_cnt = 0
    review_count = 0
    no_review_cnt = 0

    for product in products:

        # Top100 상품마다의 상품평 개수(나중에 불러오는 정보라 상품링크 안에서는 xpath로 바로 가져올 수 없음)
        try:
            review_count = int(response.xpath('//*[@id="{}"]/a/dl/dd/div/div[4]/div[2]/span[2]/text()'.format(product)).extract()[0].strip("()"))
            top100_review_total_cnt += review_count

        except Exception as e:
            no_review_cnt += 1

        # product for문 안에서 수행하는 코드
        def make_url(product, review_count):
            page_count = (review_count//100)+1
            review_urls = []
            for page in range(1, page_count+1):
                review_urls.append('http://www.coupang.com/vp/product/reviews?productId={}&page={}&size=100&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=2&ratingSummary=true'.format(product, page))
                # size는 100이 최대로 설정되어 있음
            return review_urls


        # Top100 상품마다의 크롤링할 url(리뷰 크롤링은 '상품평 보기' 클릭 후 나타나는 reviews Request URL에서 해야함)
#         review_urls[product] = make_url(product, review_count)
        review_url = make_url(product, review_count) # 리스트로 반환됨

        for url in review_url:

            review_url = url
            review_rep = requests.get(review_url)
            review_response = TextResponse(review_rep.url, body=review_rep.text, encoding='utf-8')

            # Top100 상품마다의 상품평 개수
            review_total_cnt = int(review_response.xpath('/html/body/div[2]/@data-review-total-count').extract()[0])

            # Top100 상품마다 한 page의 상품평 개수
            review_url_cnt = int(float(review_response.xpath('count(/html/body/article)').extract()[0]))

            if review_url_cnt < 100:
                for i in range(1, review_url_cnt+1): # 한개의 review_url에서 100개를 훑고 그 다음 review_url로 넘어가면 됨
                    item_nbr = product

                    date = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[2]/text()'.format(i)).extract()[0]

                    name = review_response.xpath('/html/body/article[{}]/div[1]/div[2]/span/text()'.format(i)).extract()[0][:-1]

                    item_option_ls = review_response.xpath('/html/body/article[{}]/div[1]/div[4]/text()'.format(i)).extract()[0].split(",")
                    if len(item_option_ls) == 1:
                        item = item_option_ls[0]

                    else:
                        item = item_option_ls[0]
                        option = item_option_ls[1]

                    if review_response.xpath('/html/body/article[{}]/div[3]/@class'.format(i)).extract()[0] == 'sdp-review__article__list__headline':
                        re_title = review_response.xpath('/html/body/article[{}]/div[3]/text()'.format(i)).extract()
                        review = review_response.xpath('/html/body/article[{}]/div[4]/div/text()'.format(i)).extract()

                    else:
                        re_title = ['.']
                        review = review_response.xpath('/html/body/article[{}]/div[3]/div/text()'.format(i)).extract()

                    rating = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[1]/div/@data-rating'.format(i)).extract()[0]

                    df.loc[len(df)] = {'rating':rating, 'item_nbr':item_nbr, 'item':item, 'option':option, 'date':date, 'name':name, 're_title':re_title, 'review':review}

            else:
                for i in range(1, 101):
                    item_nbr = product

                    date = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[2]/text()'.format(i)).extract()[0]

                    name = review_response.xpath('/html/body/article[{}]/div[1]/div[2]/span/text()'.format(i)).extract()[0][:-1]

                    item_option_ls = review_response.xpath('/html/body/article[{}]/div[1]/div[4]/text()'.format(i)).extract()[0].split(",")
                    if len(item_option_ls) == 1:
                        item = item_option_ls[0]

                    else:
                        item = item_option_ls[0]
                        option = item_option_ls[1]

                    if review_response.xpath('/html/body/article[{}]/div[3]/@class'.format(i)).extract()[0] == 'sdp-review__article__list__headline':
                        re_title = review_response.xpath('/html/body/article[{}]/div[3]/text()'.format(i)).extract()
                        review = review_response.xpath('/html/body/article[{}]/div[4]/div/text()'.format(i)).extract()

                    else:
                        re_title = ['.']
                        review = review_response.xpath('/html/body/article[{}]/div[3]/div/text()'.format(i)).extract()

                    rating = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[1]/div/@data-rating'.format(i)).extract()[0]

                    df.loc[len(df)] = {'rating':rating, 'item_nbr':item_nbr, 'item':item, 'option':option, 'date':date, 'name':name, 're_title':re_title, 'review':review}

    print("{}개 상품 중 review가 없는 상품 수 : {}".format(item_count_end-item_count_start, no_review_cnt))


    df['re_title_filtered'] = df['re_title'].apply(__re_title_filter)
    df['review_filtered'] = df['review'].apply(__review_filter)
    df['full_review'] = df['re_title_filtered']+df['review_filtered']
    df['pos'] = df['full_review'].apply(kkma_pos)
    df['pos_filtered'] = df['pos'].apply(__kkma_pos_filter)
    df['rating_filtered'] = df['rating'].apply(__rating_filter)

    # 'pos_filtered' == 0인 데이터 삭제
    idx = []
    for i in range(len(df)):
        if len(df['pos_filtered'][i][0]) == 0: # 리스트 안의 문자열의 길이가 0인지를 확인
            idx.append(i)
    df = df.drop(index=idx).reset_index()

    print("{}'s Top{} Review Total Count : {}".format(keyword, item_count_end, top100_review_total_cnt))
    print("{}'s Top{} Filtered Review Count : {}".format(keyword, item_count_end, len(df)))

    df_fin = df[['rating', 'rating_filtered', 'item_nbr', 'item', 'option', 'date', 'name', 'full_review', 'pos', 'pos_filtered']]

    return df_fin

def make_pos_voca(df_train, keyword, raworcsv='csv'):
    """
    train 데이터에서 단어와 pos가 담긴 vocabulary 만들기
    csv를 불러온 데이터의 경우 반드시 pos_new 컬럼이 존재해야함
    """
    pos_voca = []
    if raworcsv == 'csv':
        for i in range(len(df_train)):
            for ii in range(len(df_train["pos_new"][i])):
                if df_train["pos_new"][i][ii] not in pos_voca:
                    pos_voca.append(df_train["pos_new"][i][ii])
        pickle.dump(pos_voca, open("save_pos_voca_pkl/pos_voca_{}.pkl".format(keyword), "wb"))
        return pos_voca
    else:
        for i in range(len(df_train)):
            for ii in range(len(df_train["pos"][i])):
                if df_train["pos"][i][ii] not in pos_voca:
                    pos_voca.append(df_train["pos"][i][ii])
        pickle.dump(pos_voca, open("save_pos_voca_pkl/pos_voca_{}.pkl".format(keyword), "wb"))
        return pos_voca

def read_pos_voca(keyword):
    pos_voca = pickle.load(open("save_pos_voca_pkl/pos_voca_{}.pkl".format(keyword), "rb"))
    return pos_voca

def test_overview(product):

    reveiw_cnt_url = 'http://www.coupang.com/vp/product/reviews?productId={}&page=1&size=100&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=2&ratingSummary=true'.format(product)
    reveiw_cnt_rep = requests.get(reveiw_cnt_url)
    reveiw_cnt_response = TextResponse(reveiw_cnt_rep.url, body=reveiw_cnt_rep.text, encoding='utf-8')

    item_name = reveiw_cnt_response.xpath('/html/body/article[1]/div[1]/div[4]/text()').extract()[0].split(',')[0]
    review_total_cnt = int(reveiw_cnt_response.xpath('/html/body/div[2]/@data-review-total-count').extract()[0])

    return item_name, review_total_cnt

def df_test_len(df_test):
    return len(df_test)

def make_df_test(product):

    # 상품마다의 링크를 불러오기에 앞서 아이템 Top36 리스트 불러오기
#     products = json.loads(response.xpath('//*[@id="productList"]/@data-products').extract()[0])['indexes'][:36]

    df = pd.DataFrame(columns=['rating', "item_nbr", "item", "option", "date", "name", "re_title", 'review'])

    reveiw_cnt_url = 'http://www.coupang.com/vp/product/reviews?productId={}&page=1&size=100&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=2&ratingSummary=true'.format(product)
    reveiw_cnt_rep = requests.get(reveiw_cnt_url)
    reveiw_cnt_response = TextResponse(reveiw_cnt_rep.url, body=reveiw_cnt_rep.text, encoding='utf-8')

    item_name = reveiw_cnt_response.xpath('/html/body/article[1]/div[1]/div[4]/text()').extract()[0].split(',')[0]
    review_total_cnt = int(reveiw_cnt_response.xpath('/html/body/div[2]/@data-review-total-count').extract()[0])

    def make_url(product, review_total_cnt):
        page_count = (review_total_cnt//100)+1
        review_urls = []
        for page in range(1, page_count+1):
            review_urls.append('http://www.coupang.com/vp/product/reviews?productId={}&page={}&size=100&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=2&ratingSummary=true'.format(product, page))
            # size는 100이 최대로 설정되어 있음
        return review_urls

    review_url = make_url(product, review_total_cnt) # 리스트로 반환됨

    for url in review_url:

        review_url = url
        review_rep = requests.get(review_url)
        review_response = TextResponse(review_rep.url, body=review_rep.text, encoding='utf-8')

        # Top36 상품마다의 상품평 개수
        review_total_cnt = int(review_response.xpath('/html/body/div[2]/@data-review-total-count').extract()[0])

        # Top36 상품마다 한 page의 상품평 개수
        review_url_cnt = int(float(review_response.xpath('count(/html/body/article)').extract()[0]))

        if review_url_cnt < 100:
            for i in range(1, review_url_cnt+1): # 한개의 review_url에서 100개를 훑고 그 다음 review_url로 넘어가면 됨
                item_nbr = product

                date = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[2]/text()'.format(i)).extract()[0]

                name = review_response.xpath('/html/body/article[{}]/div[1]/div[2]/span/text()'.format(i)).extract()[0][:-1]

                item_option_ls = review_response.xpath('/html/body/article[{}]/div[1]/div[4]/text()'.format(i)).extract()[0].split(",")
                if len(item_option_ls) == 1:
                    item = item_option_ls[0]

                else:
                    item = item_option_ls[0]
                    option = item_option_ls[1]

                if review_response.xpath('/html/body/article[{}]/div[3]/@class'.format(i)).extract()[0] == 'sdp-review__article__list__headline':
                    re_title = review_response.xpath('/html/body/article[{}]/div[3]/text()'.format(i)).extract()
                    review = review_response.xpath('/html/body/article[{}]/div[4]/div/text()'.format(i)).extract()

                else:
                    re_title = ['.']
                    review = review_response.xpath('/html/body/article[{}]/div[3]/div/text()'.format(i)).extract()

                rating = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[1]/div/@data-rating'.format(i)).extract()[0]

                df.loc[len(df)] = {'rating':rating, 'item_nbr':item_nbr, 'item':item, 'option':option, 'date':date, 'name':name, 're_title':re_title, 'review':review}

        else:
            for i in range(1, 101):
                item_nbr = product

                date = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[2]/text()'.format(i)).extract()[0]

                name = review_response.xpath('/html/body/article[{}]/div[1]/div[2]/span/text()'.format(i)).extract()[0][:-1]

                item_option_ls = review_response.xpath('/html/body/article[{}]/div[1]/div[4]/text()'.format(i)).extract()[0].split(",")
                if len(item_option_ls) == 1:
                    item = item_option_ls[0]

                else:
                    item = item_option_ls[0]
                    option = item_option_ls[1]

                if review_response.xpath('/html/body/article[{}]/div[3]/@class'.format(i)).extract()[0] == 'sdp-review__article__list__headline':
                    re_title = review_response.xpath('/html/body/article[{}]/div[3]/text()'.format(i)).extract()
                    review = review_response.xpath('/html/body/article[{}]/div[4]/div/text()'.format(i)).extract()

                else:
                    re_title = ['.']
                    review = review_response.xpath('/html/body/article[{}]/div[3]/div/text()'.format(i)).extract()

                rating = review_response.xpath('/html/body/article[{}]/div[1]/div[3]/div[1]/div/@data-rating'.format(i)).extract()[0]

                df.loc[len(df)] = {'rating':rating, 'item_nbr':item_nbr, 'item':item, 'option':option, 'date':date, 'name':name, 're_title':re_title, 'review':review}

    df['re_title_filtered'] = df['re_title'].apply(__re_title_filter)
    df['review_filtered'] = df['review'].apply(__review_filter)
    df['full_review'] = df['re_title_filtered']+df['review_filtered']
    df['pos'] = df['full_review'].apply(kkma_pos)
    df['pos_filtered'] = df['pos'].apply(__kkma_pos_filter)
    df['rating_filtered'] = df['rating'].apply(__rating_filter)

    # 'pos_filtered' == 0인 데이터 삭제
    idx = []
    for i in range(len(df)):
        if len(df['pos_filtered'][i][0]) == 0: # 리스트 안의 문자열의 길이가 0인지를 확인
            idx.append(i)
    df = df.drop(index=idx).reset_index()

    print("{}'s Review Total Count : {}".format(item_name, review_total_cnt))
    print("{}'s Filtered Review Count : {}".format(item_name, len(df)))

    df_fin = df[['rating', 'rating_filtered', 'item_nbr', 'item', 'option', 'date', 'name', 'full_review', 'pos', 'pos_filtered']]

    return df_fin

def __re_title_filter(a):
    for i in range(len(a[0].split('\n'))):
        if len(a[0].split('\n')[i].strip()) != 0:
            return a[0].split('\n')[i].strip()
        else:
            return '.'

def __review_filter(a):
    s = ''
    for i in range(len(a)):
        for ii in range(len(a[i].split('\n'))):
            if a[i].split('\n')[ii].strip() != 0:
                s += a[i].split('\n')[ii].strip() + ' '
    return s

def kkma_pos(a):
    kkma = Kkma()
    return kkma.pos(a)

def __kkma_pos_filter(a):
    """
    df['pos']를 받아 품사 필터링을 거쳐 해당 키워드 리스트 반환
    """
    pos_ls = ['NNG','NNP','VA','UN','XR','MAG','ECE']
    ls = []
    s = ""
    for i in range(len(a)):
        #NNG(보통 명사), NNP(고유 명사), VA(형용사), XR(어근), MAG(일반 부사), UN(명사추정범주)
        if a[i][1] in pos_ls:
            s += a[i][0] + ','
    ls.append(s)
    return ls

def __rating_filter(a):
    if a == '3' or a == '4' or a == '5':
        return 1
        # 긍정
    else:
        return 0
        # 부정

def Vectorizer_train(df_train, keyword, stop_words=None):
    """
    df_train의 'pos_filtered' column을 받아 vectorized df를 생성해주는 함수
    """
    train_corpus = []
    for i in range(len(df_train['pos_filtered'])):
        train_corpus.append(df_train['pos_filtered'][i][0])

    vect = CountVectorizer(token_pattern=r"\b\w+\b", stop_words=stop_words) # 한 글자도 corpus에 포함될 수 있게 해주는 정규표현식
    vect.fit(train_corpus)

    pickle.dump(vect, open("save_vect_pkl/vect_{}.pkl".format(keyword), "wb"))

    vect_ls = []
    for i in range(len(df_train)):
        vect_ls.append(vect.transform(df_train['pos_filtered'][i]).toarray()[0])
        # 리스트안의 리스트로 반환되기 때문에 vect_ls 안에 append하기 전에 [0]로 꺼내줌

    # 문자열 하나를 문장으로 간주
    # corpus에는 여러 문장이 하나의 리스트 안의 문자열 요소로 들어가야 하고, ['an apple is red', 'the boy is young']
    # transform할때는 한 문장이 하나의 리스트 안의 문자열 요소로 들어가야 함 ['an apple is red']

    df_vec = pd.DataFrame(vect_ls)

    return df_vec

def Vectorizer_test(df_input, keyword):
    """
    df_input의 'pos_filtered' column을 받아 vectorized df를 생성해주는 함수
    df와 vect.vocabulary_를 반환
    """
    vect = pickle.load(open("save_vect_pkl/vect_{}.pkl".format(keyword), "rb"))

    vect_ls = []
    for i in range(len(df_input)):
        vect_ls.append(vect.transform(df_input['pos_filtered'][i]).toarray()[0])
        # 리스트안의 리스트로 반환되기 때문에 vect_ls 안에 append하기 전에 [0]로 꺼내줌

    # 문자열 하나를 문장으로 간주
    # corpus에는 여러 문장이 하나의 리스트 안의 문자열 요소로 들어가야 하고, ['an apple is red', 'the boy is young']
    # transform할때는 한 문장이 하나의 리스트 안의 문자열 요소로 들어가야 함 ['an apple is red']

    df_vec = pd.DataFrame(vect_ls)

    return df_vec, vect.vocabulary_

def top_word_negative(X_test, y_test_pred, df_test, pos_voca, voca, word_cnt=20):
    """
    pos_voca에는 단어와 pos 정보가 담겨있고
    voca에는 단어와 vect모델 안에서의 단어별 index 정보가 담겨있다
    """
    y_test_pred = pd.DataFrame(y_test_pred, columns=['prediction'])
    pred_n_test = pd.concat([y_test_pred, X_test], axis=1)
    top_word = pd.DataFrame(pred_n_test[pred_n_test['prediction']==0].sum().sort_values(ascending=False), columns=['count']).reset_index()
    # reset_index()하면 원래 series에서 갖고있던 index를 value로 만들어줌

    top_word = top_word[top_word['count'] > 0]

    def match_word(a):
        for key, value in voca.items():
            if value == a:
                return key

    def real_pos(a):
        for i in range(len(pos_voca)):
            if a == pos_voca[i][0]:
                return pos_voca[i][1]

    top_word['word'] = top_word['index'].apply(match_word)
    top_word['pos'] = top_word['word'].apply(real_pos)

    top_word = top_word[top_word['pos'].isin(['NNG','NNP', 'XR', 'UN'])].reset_index()[:word_cnt]

    def review_return(a):
        idx_ls = list(pred_n_test[(pred_n_test['prediction']==0)&(pred_n_test[a] >= 1)].index)
        return list(df_test[df_test.index.isin(idx_ls)]['full_review'].values)

    top_word['review'] = top_word['index'].apply(review_return)

    return top_word[['index','word', 'pos', 'count', 'review']]


def top_word_positive(X_test, y_test_pred, df_test, pos_voca, voca, word_cnt=20):
    """
    pos_voca에는 단어와 pos 정보가 담겨있고
    voca에는 단어와 vect모델 안에서의 단어별 index 정보가 담겨있다
    """
    y_test_pred = pd.DataFrame(y_test_pred, columns=['prediction'])
    pred_n_test = pd.concat([y_test_pred, X_test], axis=1)
    top_word = pd.DataFrame(pred_n_test[pred_n_test['prediction']==1].sum().sort_values(ascending=False), columns=['count']).reset_index()
    # reset_index()하면 원래 series에서 갖고있던 index를 value로 만들어줌

    top_word_ = top_word[top_word['index'] != 'prediction'].reset_index(drop=True)
    # prediction이 1이기 때문에 count값이 정수로 생기고, 그 row를 삭제해줌
    top_word = top_word[top_word['count'] > 0]

    def match_word(a):
        for key, value in voca.items():
            if value == a:
                return key

    def real_pos(a):
        for i in range(len(pos_voca)):
            if a == pos_voca[i][0]:
                return pos_voca[i][1]

    top_word['word'] = top_word['index'].apply(match_word)
    top_word['pos'] = top_word['word'].apply(real_pos)

    top_word = top_word[top_word['pos'].isin(['NNG','NNP', 'XR', 'UN'])].reset_index()[:word_cnt]

    def review_return(a):
        idx_ls = list(pred_n_test[(pred_n_test['prediction']==1)&(pred_n_test[a] >= 1)].index)
        return list(df_test[df_test.index.isin(idx_ls)]['full_review'].values)

    top_word['review'] = top_word['index'].apply(review_return)

    return top_word[['index','word', 'pos', 'count', 'review']]

def top_word_word(df_top_word):
    return list(df_top_word['word'].values)

def top_word_count(df_top_word):
    # jsonify에서 np.int는 오류나므로 int로 형변환해주어야 함
    return [int(val) for val in df_top_word['count'].values]

def top_word_review(df_top_word):
    return list(df_top_word['review'].values)

def neg_or_pos(y_test_pred):
    """
    y_test_pred를 받아서 negative 댓글 개수와 positive 댓글 개수를 int로 반환
    """
    y_test_pred = pd.DataFrame(y_test_pred, columns=['prediction'])
    neg_or_pos = y_test_pred.groupby(by='prediction').size().reset_index(name='count')
    neg = int(list(neg_or_pos[neg_or_pos['prediction']==0]['count'].values)[0])
    pos = int(list(neg_or_pos[neg_or_pos['prediction']==1]['count'].values)[0])
    return neg, pos
