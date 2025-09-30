import streamlit as st
import requests
import json
from typing import List, Dict

st.set_page_config(page_title='RAG HR Chatbot', layout='centered')

st.title('HR Policy Q&A — RAG Chatbot')
st.write('Ask questions about the HR policy PDF. This app queries the local backend and shows provenance.')

# Small helper
def query_backend(q: str, top_k: int = 5) -> Dict:
    url = 'http://127.0.0.1:8000/query'  # use loopback explicitly
    payload = {'q': q, 'top_k': top_k}
    headers = {'Content-Type': 'application/json'}
    resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()

with st.sidebar:
    st.header('Settings')
    top_k = st.number_input('Top K results', min_value=1, max_value=20, value=5)
    show_embeddings = st.checkbox('Show embeddings (large)', value=False)

st.markdown('---')

col1, col2 = st.columns([3,1])
with col1:
    query_text = st.text_input('Enter your HR question', value='How many earned leaves do I get?')
with col2:
    if st.button('Ask'):
        st.session_state['trigger'] = True

# allow pressing Enter to trigger a run
trigger = st.button('Run query (press Enter after typing)')

if 'trigger' not in st.session_state:
    st.session_state['trigger'] = False

# run when user triggers either button or Enter
if st.session_state['trigger'] or trigger:
    if not query_text or not query_text.strip():
        st.error('Please type a question first.')
    else:
        st.session_state['trigger'] = False
        with st.spinner('Querying backend...'):
            try:
                result = query_backend(query_text, top_k=int(top_k))
            except requests.exceptions.RequestException as re:
                st.error(f'Backend request failed: {re}')
            except Exception as e:
                st.error(f'Unexpected error: {e}')
            else:
                # show the synthesized answer (or concatenated snippets)
                st.subheader('Answer')
                answer = result.get('answer') or 'No answer returned.'
                st.write(answer)

                st.subheader('Sources (retrieved chunks)')
                sources = result.get('sources') or []
                if not sources:
                    st.info('No chunks returned.')
                else:
                    for s in sources:
                        st.markdown(f"**Page {s.get('page', 'N/A')} — score {s.get('score', 0):.4f}**")
                        st.write(s.get('text', '')[:1000])
                        if show_embeddings:
                            st.write('Embedding length:', len(s.get('embedding', [])))
                st.markdown('---')

# small footer
st.caption('Local demo — backed by a FAISS/NumPy index. Backend: http://127.0.0.1:8000')
