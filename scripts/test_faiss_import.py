import traceback, sys
try:
    from index.faiss_utils import FAISSWrapper
    w = FAISSWrapper()
    print('FAISSWrapper instantiated OK')
    print('Index size:', w.index_size())
    import numpy as np
    q = np.zeros(w.embeddings.shape[1], dtype=np.float32)
    res = w.search(q, top_k=1)
    print('Sample result keys:', list(res[0].keys()) if res else 'no results')
except Exception:
    traceback.print_exc()
    sys.exit(1)
