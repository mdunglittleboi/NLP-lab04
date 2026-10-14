import re
import numpy as np

_HTML  = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]")   # giữ lại don't, can't, và dấu ! ?

def clean_text(text: str) -> str:
    # lowercase, bỏ thẻ HTML (IMDb có nhiều '<br />'), chuẩn hoá khoảng trắng
    text = _HTML.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str):
    return _TOKEN.findall(clean_text(text))

def avg_vector(tokens, wv, dim):
    # Trung bình vector của các từ có trong từ điển embedding; không có từ nào -> vector 0
    vecs = [wv[w] for w in tokens if w in wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

def docs_to_matrix(token_lists, wv):
    dim = wv.vector_size
    return np.vstack([avg_vector(t, wv, dim) for t in token_lists])
