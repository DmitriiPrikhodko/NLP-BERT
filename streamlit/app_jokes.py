import os, glob, time
import streamlit as st
import torch
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL_ID = "sberbank-ai/rugpt3small_based_on_gpt2"
ADAPTER_DIR   = os.getenv("ADAPTER_DIR", str(REPO_ROOT / "weights"))  # -> NLP-BERT/weights
USE_BF16      = True


st.set_page_config(page_title="", page_icon="✨", layout="wide")
st.title("🤡 Генерация шуток про Айти")

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_adapter_dir(root: Optional[str]) -> Optional[str]:
    if not root:
        return None
    if os.path.isdir(root) and os.path.isfile(os.path.join(root, "adapter_config.json")):
        return root
    if os.path.isdir(root):
        for p in sorted(glob.glob(os.path.join(root, "checkpoint-*")), reverse=True):
            if os.path.isfile(os.path.join(p, "adapter_config.json")):
                return p
    return root 

@st.cache_resource(show_spinner=True)
def load_pipeline(base_id: str, adapter_hint: Optional[str], bf16: bool):
    dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype).to(device)
    base.config.use_cache = True

    model = base
    if adapter_hint:
        target = find_adapter_dir(adapter_hint)
        try:
            model = PeftModel.from_pretrained(base, target).to(device)
        except Exception as e:
            st.error(f"Не удалось загрузить LoRA из '{target}': {e}")
            st.stop()
    model.eval()
    return tok, model

tok, model = load_pipeline(BASE_MODEL_ID, ADAPTER_DIR, USE_BF16)


col_left, col_right = st.columns([3, 1])

with col_left:
    prompt = st.text_area(
        "Введите prompt / начало текста",
        height=180,
        placeholder="Например: Расскажи короткий анекдот про программистов...",
        label_visibility="visible",
    )
    go = st.button("Пошутить", type="primary", use_container_width=True)

with col_right:
    max_new_tokens = st.slider("Длина", 10, 1024, 160, 10)
    num_return_sequences = st.slider("Вариантов", 1, 8, 3, 1)
    do_sample = st.checkbox("Стохастика", value=True)
    temperature = st.slider("temperature", 0.1, 1.5, 0.9, 0.05)
    top_p = st.slider("top-p", 0.5, 1.0, 0.95, 0.01)
    top_k = st.slider("top-k", 0, 500, 50, 10)
    repetition_penalty = st.slider("repetition_penalty", 1.0, 1.5, 1.1, 0.01)
    no_repeat_ngram_size = st.slider("no_repeat_ngram_size", 0, 10, 3, 1)
    seed = st.number_input("seed", value=42, step=1)
    stop_token = st.text_input("Стоп-токен", value="")

def generate_many(
    prompt: str,
    n: int,
    max_new: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    stop_token: Optional[str],
    seed: Optional[int] = None,
) -> List[str]:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=int(max_new),
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if no_repeat_ngram_size and int(no_repeat_ngram_size) > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if do_sample and top_k and int(top_k) > 0:
        gen_kwargs["top_k"] = int(top_k)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    outs = []
    for i in range(n):
        out_ids = model.generate(**inputs, **gen_kwargs)
        text_full = tok.decode(out_ids[0], skip_special_tokens=True)
        gen_part = text_full[len(prompt):]
        if stop_token:
            idx = gen_part.find(stop_token)
            if idx != -1:
                gen_part = gen_part[:idx]
        outs.append(gen_part.strip())
        if seed is not None and do_sample:
            torch.manual_seed(int(seed) + i + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + i + 1)
    return outs

if go:
    if not prompt.strip():
        st.warning("Введите prompt.")
    else:
        t0 = time.perf_counter()
        with st.spinner("Начинаю шутить…"):
            results = generate_many(
                prompt=prompt,
                n=num_return_sequences,
                max_new=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_token=stop_token if stop_token.strip() else None,
                seed=int(seed),
            )
        elapsed = time.perf_counter() - t0


        st.subheader("Результаты")
        for i, txt in enumerate(results, 1):
            st.markdown(f"**Вариант {i}**")
            st.write(txt)
            st.divider()


        char_lens = [len(r) for r in results]
        tok_lens = [len(tok(r, add_special_tokens=False)["input_ids"]) for r in results]
        avg_chars = sum(char_lens) / len(char_lens)
        avg_tokens = sum(tok_lens) / len(tok_lens)

        st.markdown("#### Краткий отчёт")
        st.markdown(
            f"""
- Вариантов: **{len(results)}**
- Параметры: `max_new_tokens={max_new_tokens}`, `do_sample={do_sample}`, `temperature={temperature}`, `top_p={top_p}`, `top_k={top_k}`, `repetition_penalty={repetition_penalty}`, `no_repeat_ngram_size={no_repeat_ngram_size}`, `seed={int(seed)}`
- Средняя длина ответа: **{avg_chars:.0f} символов** (~{avg_tokens:.0f} токенов)
- Время генерации: **{elapsed:.2f} с** на устройство **{device}**
- Модель: **{BASE_MODEL_ID}**, LoRA: **{find_adapter_dir(ADAPTER_DIR)}**
            """
        )


st.divider()
st.header("О проекте")
st.markdown("""
Этот проект — демо генератора шуток про IT на базе ruGPT3small.

**Что внутри:**
- дообучение на корпусе шуток про айтишников (порядка 2000),
- интерфейс с контролем длины/креативности ответов,
- поддержка нескольких вариантов выдачи.
            
""")
