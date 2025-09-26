import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(layout="wide")

st.title("Оценка степени токсичности пользовательского сообщения")

st.markdown(
    """
    Модель [rubert-tiny-toxicity](https://huggingface.co/cointegrated/rubert-tiny-toxicity) оценивает вероятность, на сколько текст является токсичным или опасным.
    
Сообщение оценивается по 5 классам:    
* Адекватный (non-toxic): текст не содержит оскорблений, нецензурной лексики и угроз (в смысле, используемом в соревновании [OK ML Cup](https://cups.mail.ru/ru/tasks/1048)).
* Оскорбительный (insult): осдержит оскорбления.
* Ругательства (obscenity): содержит нецензурную лексику.
* Угрозы (threat): содержит угрозы.
* Опасный (dangerous): текст является неприемлемым (в смысле, определённом в работе [Babakov и др.](https://arxiv.org/abs/2103.05345), то есть он может нанести вред репутации автора.
 
Текст можно считать безопасным, если он одновременно является non-toxic и не dangerous.

Уровень токсичности - аггрегированный показатель, рассчитываесят по формуле 1 - probs[0] * (1 - probs[-1]) - Инверсия вероятности, что текст одновременно и адекватный, и не опасный.
- probs[0] → вероятность класса non-toxic 
- probs[-1] → вероятность класса dangerous

"""
)
st.markdown("---")
# ==== Загрузка модели ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)

# ==== Функция предсказания ====
def text2toxicity(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()

    class_names = ["Адекватный", "Оскорбительный", "Ругательства", "Угрозы", "Опасный"]

    results = []
    for probs in proba:
        agg_value = 1 - probs[0] * (1 - probs[-1])
        lines = [f"**Уровень токсичности:** {agg_value * 100:.2f}%"]
        for i, name in enumerate(class_names):
            lines.append(f"- Класс {i} ({name}): {probs[i]:.4f}")
        results.append("\n".join(lines))
    return "\n\n".join(results)

# ==== Интерфейс Streamlit ====
st.subheader("Введите сообщение для проверки")

# Хранение истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовка истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Введите сообщение для проверки"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # показываем сообщение пользователя
    with st.chat_message("user"):
        st.markdown(prompt)

    # предсказание модели
    result = text2toxicity(prompt)

    # ответ модели
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})