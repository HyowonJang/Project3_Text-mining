from flask import *
import pickle
import numpy as np
import function_front as f
from sklearn.metrics import f1_score

app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True,
)
# 서버를 내렸다 올려주지 않아도 반영이 됨

models = {}

def init():
    with open("save_model_pkl/model_EHD_0_100_13806.pkl", "rb") as f:
        models["classification"] = pickle.load(f)

# models - global variable
# models["classification"] - key
# pickle.load - value

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    result = {"status":200}

    # classification_list = ["정치","경제","사회","생활","문화","세계","IT/과학"]
    model = models["classification"]
    # init함수에서 predict함수로 model을 넘겨주기 위해 models["classification"]로 저장..

    product = request.values.get("link")
    # /predict? 다음의 'link'라는 key값에 해당하는 value를 받음
    item_name, review_total_cnt = f.test_overview(product)
    result["item_name"] = item_name
    result["review_total_cnt"] = review_total_cnt

    df_test = f.make_df_test(product)
    result["valid_review_cnt"] = f.df_test_len(df_test)

    X_test, voca = f.Vectorizer_test(df_test, 'EHD')
    y_test = df_test['rating_filtered']

    y_test_pred = model.predict(X_test)
    pos_voca = f.read_pos_voca('EHD')
    # 추후 외장하드 외 다른 키워드 추가하면 변수화할 것

    result["f1_score"] = f1_score(y_test, y_test_pred, average='weighted')

    result["neg_or_pos_cnt"] = list(f.neg_or_pos(y_test_pred))
    result["neg_or_pos"] = ["부정","긍정"]

    # result["neg_result"] = {
    #     "word" : ["word1", "word2"],
    #     "count" : [np.int64(0), 5],
    #     "review" : [["sd", "Ss"], ["a", " dd"]]
    # }


    df_top_word_negative = f.top_word_negative(X_test, y_test_pred, df_test, pos_voca, voca, word_cnt=5)

    result["neg_result"] = {
        "word" : f.top_word_word(df_top_word_negative),
        "count" : f.top_word_count(df_top_word_negative),
        "review" : f.top_word_review(df_top_word_negative)
    }

    df_top_word_positive = f.top_word_positive(X_test, y_test_pred, df_test, pos_voca, voca, word_cnt=5)

    result["pos_result"] = {
        "word" : f.top_word_word(df_top_word_positive),
        "count" : f.top_word_count(df_top_word_positive),
        "review" : f.top_word_review(df_top_word_positive)
    }

# pip install gunicorn
# gunicorn --reload article:app


    return jsonify(result)

# terminal에서 python article.py 실행

init()
app.run()

# python article.py
