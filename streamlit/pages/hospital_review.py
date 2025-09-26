import streamlit as st
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import os
import sys
import requests
from nltk.corpus import stopwords
from catboost import CatBoostClassifier
from main import ROOT_DIR
import re
import string

stop_words = set(stopwords.words("russian"))
stop_words.difference_update({"не", "нет", "без"})
ROOT_DIR = os.path.abspath(os.path.dirname("main.py"))
# st.write(ROOT_DIR)
CURR_DIR = os.path.dirname(__file__)
# st.write(CURR_DIR)

module_path = os.path.join(ROOT_DIR, "src")
sys.path.append(module_path)
# print(ROOT_DIR)
from LSTMattention_classes import Config, BahdanauAttention, LSTMBahdanauAttention
from Bert_class import MyPersonalTinyBert, BertInputs

DEVICE = torch.device("cpu")
# =====================
# 1) ЗАГРУЗКА МОДЕЛЕЙ
# =====================

# ---- CatBoost + TF-IDF ----
tfidf = joblib.load(
    os.path.join(
        ROOT_DIR,
        "weights",
        "Catboost+TFIDF",
        "tfidf_vectorizer.pkl",
    )
)  # <-- подставь путь
catboost_model = CatBoostClassifier()
catboost_model.load_model(
    os.path.join(
        ROOT_DIR,
        "weights",
        "Catboost+TFIDF",
        "catboost_model.cbm",
    )
)  # <-- подставь путь


# Загружаем веса LSTM (заглушка)


lstm_vocab = joblib.load(
    os.path.join(
        ROOT_DIR,
        "weights",
        "W2V_weights",
        "W2V_model.model",
    )
)  # <-- подставь путь
vocab_size = len(lstm_vocab.wv) + 1
lstm_config = Config(
    n_layers=4,
    embedding_size=64,
    hidden_size=32,
    vocab_size=vocab_size,
    device=DEVICE,
    seq_len=64,
    bidirectional=False,
)
lstm_model = LSTMBahdanauAttention(lstm_config)
lstm_model.load_state_dict(
    torch.load(
        os.path.join(
            ROOT_DIR,
            "weights/",
            "LSTM",
            "weight_epoch_5.pth",
        ),
        map_location="cpu",
    )
)
lstm_model.eval()

# ---- RuBERT-tiny2 + классификатор ----

RUBERT_WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights", "RuBert")
RUBERT_WEIGHTS_PATH = os.path.join(RUBERT_WEIGHTS_DIR, "weights_RuBert.pth")
os.makedirs(RUBERT_WEIGHTS_DIR, exist_ok=True)
if not os.path.exists(RUBERT_WEIGHTS_PATH):
    public_url = "https://disk.yandex.ru/d/OcbCPfbaaa8dbg"
    r = requests.get(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download",
        params={"public_key": public_url},
    )
    r.raise_for_status()
    download_url = r.json()["href"]
    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        with open(RUBERT_WEIGHTS_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

rubert_classifier = MyPersonalTinyBert()
rubert_classifier.load_state_dict(
    torch.load(os.path.join(RUBERT_WEIGHTS_PATH), map_location="cpu")
)

rubert_classifier.eval()


# =====================
# 2) ПРЕДОБРАБОТКА
# =====================
def preprocess(text, max_len=50):
    text = text.lower()
    text = re.sub("<.*?>", "", text)  # html tags
    text = "".join(
        [c for c in text if c not in string.punctuation]
    )  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([word for word in text.split() if not word.isdigit()])
    return text


def get_words_by_freq(sorted_words: list[tuple[str, int]], n: int = 10) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))


def padding(review_int: list, seq_len: int) -> np.array:  # type: ignore
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[:seq_len]
        features[i, :] = np.array(new)

    return features


def preprocess_lstm(input_string: str, seq_len: int, vocab, verbose: bool = False):
    """Function for all preprocessing steps on a single string

    Args:
        input_string (str): input single string for preprocessing
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros
        vocab_to_int (dict, optional): word corpus {'word' : int index}. Defaults to vocab_to_int.

    Returns:
        list: preprocessed string
    """
    result_list = []
    preprocessed_string = preprocess(input_string)
    vocab_to_int = vocab.wv.key_to_index
    # indices = [word_to_index[word] for word in text.split() if word in word_to_index]
    for word in preprocessed_string.split():
        try:
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            if verbose:
                print(f"{e}: not in dictionary!")
            pass
    result_padded = padding([result_list], seq_len)[0]

    return torch.tensor(result_padded, dtype=torch.long)


# =====================
# 3) STREAMLIT UI
# =====================

st.title("Анализ тональности отзыва")
st.write("Введите отзыв, чтобы получить предсказания 3 разных моделей")

user_input = st.text_area("Отзыв", "")


def format_result(label, prob):
    """Возвращает текст с цветом и корректной вероятностью негативного"""
    if label == 1:  # позитив
        color = "green"
        display_prob = prob
    else:  # негатив
        color = "red"
        display_prob = 1 - prob
    return f"<span style='color:{color}; font-weight:bold'>{'Позитивный' if label==1 else 'Негативный'} (вероятность: {display_prob:.3f})</span>"


if st.button("Предсказать"):
    if not user_input.strip():
        st.warning("Введите текст!")
    else:
        # --- 1) CatBoost + TF-IDF ---
        start_time_catboost = time.time()
        tfidf_features = tfidf.transform([preprocess(user_input)])
        catboost_pred = catboost_model.predict(tfidf_features)[0]
        # Если есть вероятность, используем её, иначе выставляем dummy
        if hasattr(catboost_model, "predict_proba"):
            catboost_prob = catboost_model.predict_proba(tfidf_features)[0][1]
        else:
            catboost_prob = 1.0 if catboost_pred == 1 else 0.0
        catboost_time = time.time() - start_time_catboost

        # --- 2) LSTM + Attention ---
        start_time_lstm = time.time()
        lstm_input = preprocess_lstm(
            user_input, lstm_config.seq_len, lstm_vocab
        ).unsqueeze(0)
        with torch.no_grad():
            lstm_pred = lstm_model(lstm_input)[0].sigmoid().item()
        lstm_pred_label = 1 if lstm_pred >= 0.5 else 0
        lstm_time = time.time() - start_time_lstm

        # --- 3) RuBERT-tiny2 + классификатор ---
        start_time_rubert = time.time()
        tokens = tokenizer(
            preprocess(user_input),
            max_length=64,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # bert_input = BertInputs(tokens)
        with torch.no_grad():
            output = rubert_classifier(tokens["input_ids"], tokens["attention_mask"])
        rubert_prob = torch.sigmoid(output).item()
        rubert_pred_label = 1 if rubert_prob >= 0.5 else 0
        rubert_time = time.time() - start_time_rubert

        # =====================
        # РЕЗУЛЬТАТЫ
        # =====================
        st.subheader("Результаты предсказания:")
        st.markdown(
            f"**CatBoost + TF-IDF:** {format_result(catboost_pred, catboost_prob)} (время: {catboost_time:.3f} сек)",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**LSTM + Attention:** {format_result(lstm_pred_label, lstm_pred)} (время: {lstm_time:.3f} сек)",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**RuBERT-tiny2:** {format_result(rubert_pred_label, rubert_prob)} (время: {rubert_time:.3f} сек)",
            unsafe_allow_html=True,
        )

st.subheader("F1-score моделей")
f1_data = {
    "Модель": ["CatBoost + TF-IDF", "LSTM + Attention", "RuBERT-tiny2"],
    "F1-score": [0.91, 0.94, 0.90],
}
st.table(f1_data)
