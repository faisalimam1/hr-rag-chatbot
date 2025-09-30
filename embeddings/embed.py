# embeddings/embed.py
# Provides get_embedding(text) and batch_embed(texts, batch_size=16)
# Uses OpenAI if OPENAI_API_KEY set, otherwise sentence-transformers offline.

import os

_OPENAI_KEY = os.getenv('OPENAI_API_KEY')

if _OPENAI_KEY:
    try:
        import openai
        OPENAI = openai
        OPENAI.api_key = _OPENAI_KEY
        USE_OPENAI = True
    except Exception:
        USE_OPENAI = False
else:
    USE_OPENAI = False

if not USE_OPENAI:
    try:
        from sentence_transformers import SentenceTransformer
        _smodel = SentenceTransformer('all-MiniLM-L6-v2')
        USE_SENTENCE = True
    except Exception:
        _smodel = None
        USE_SENTENCE = False

def get_embedding(text: str):
    text = text[:8192]
    if USE_OPENAI:
        model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        resp = OPENAI.Embedding.create(model=model, input=text)
        return resp['data'][0]['embedding']
    elif USE_SENTENCE:
        vec = _smodel.encode(text)
        return vec.tolist()
    else:
        raise RuntimeError('No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers.')

def batch_embed(texts, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if USE_OPENAI:
            model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
            resp = OPENAI.Embedding.create(model=model, input=batch)
            for r in resp['data']:
                out.append(r['embedding'])
        else:
            vecs = _smodel.encode(batch)
            for v in vecs:
                out.append(v.tolist())
    return out
