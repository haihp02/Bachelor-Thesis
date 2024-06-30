import time
import streamlit as st

from indexing_and_retrieving_module.indexer import Indexer
from indexing_and_retrieving_module.retriever import Retriever

@st.cache_resource
def setup_and_load():
    # Load here to cache
    from streamlit_utils import (
        collection,
        ColBERT_model,
        ColBERT_indexer,
        ColBERTwKW_indexer,
        ColBERT_retriever,
        PLAID_ColBERT_indexer,
        PLAID_ColBERT_retriever,
        PLAID_ColBERTwKW_indexer,
        PLAID_ColBERTwKW_retriever,
        COIL_model,
        COIL_indexer,
        COIL_retriever,
        ws_model
    )
    artifacts = {
        'ColBERT': {
            'model': ColBERT_model,
            'index': ColBERT_indexer,
            'retriever': ColBERT_retriever,
            'plaid_index': PLAID_ColBERT_indexer,
            'plaid_retriever': PLAID_ColBERT_retriever
        },
        'ColBERT-Kw': {
            'model': ColBERT_model,
            'index': ColBERTwKW_indexer,
            'retriever': ColBERT_retriever,
            'plaid_index': PLAID_ColBERTwKW_indexer,
            'plaid_retriever': PLAID_ColBERTwKW_retriever
        },
        'COIL': {
            'model': COIL_model,
            'index': COIL_indexer,
            'retriever': COIL_retriever
        }
    }
    return artifacts, collection, ws_model

artifacts, collection, ws_model = setup_and_load()

def preprocess_query(query: str):
    wsed_query = ' '.join(ws_model.word_segment(query))
    return wsed_query

# Dummy function to simulate document search
def search_documents(query, model, topk, artifacts, collection, use_plaid=False):
    query = preprocess_query(query)
    index: Indexer = artifacts[model]['index'] if not use_plaid else artifacts[model]['plaid_index']
    retriever: Retriever = artifacts[model]['retriever'] if not use_plaid else artifacts[model]['plaid_retriever']
    if use_plaid:
        # Index already prepared
        retrieval_result = retriever.search(queries=query, return_sorted=True)[0][:topk]
    else:
        # Pass the Index
        retrieval_result = retriever.search(queries=query, indexer=index, return_sorted=True)[0][:topk]
    results = [collection[retrieved_doc['corpus_id']] for retrieved_doc in retrieval_result]
    return results

def main():
    st.title("Information Retrieval Demo")\
    # Sidebar for settings
    st.sidebar.header("Settings")
    model_list = ['BM25', 'vietnamese-bi-encoder', 'ColBERT', 'ColBERT-Kw', 'COIL']
    model = st.sidebar.selectbox(
        "Choose a model to use:", model_list, index=model_list.index('ColBERT'))
    topk = st.sidebar.slider("Number of top documents to return:", 1, 20, 10)
    use_plaid = False
    if model in ["ColBERT", "ColBERT-Kw"]:
        use_plaid = st.sidebar.checkbox("PLAID")
    # Input query
    query = st.text_input("Enter your query:")
    # Button to trigger search
    if st.button("Search"):
        if query and model:
            # Time the search process
            start_time = time.time()
            # Perform the search
            results = search_documents(query=query, model=model, topk=topk, artifacts=artifacts, collection=collection, use_plaid=use_plaid)
            end_time = time.time()
            search_duration = end_time - start_time

            # Display the results
            st.subheader(f"Top {topk} Most Relevant Documents:")
            st.write(f"Search completed in {search_duration:.4f} seconds")
            for i, result in enumerate(results, 1):
                st.markdown(f"""
                    <div style="padding: 15px; margin: 10px 0; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9;">
                        <h6 style="color: #333;">Document #{i}</h6>
                        <p style="color: #555;">{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Please enter a query and select a model.")

if __name__ == "__main__":
    main()