import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os
import pickle
import streamlit as st
import pandas as pd
import io

# ==============================================================================
# VectorSearch Class (Your Original Code)
# ==============================================================================

class VectorSearch:
    """
    A class for performing optimized similarity search on sentence embeddings.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.sentences = []

    def build_index(self, sentences, nlist=2):
        """Builds an optimized FAISS index from a list of sentences."""
        st.write(f"\nBuilding index for {len(sentences)} sentences...")
        self.sentences = sentences

        # 1. Create embeddings
        start_time = time.time()
        with st.spinner('Creating embeddings... This may take a moment.'):
            embeddings = self.model.encode(self.sentences, show_progress_bar=True, convert_to_numpy=True)
        st.success(f"Embeddings created in {time.time() - start_time:.2f} seconds.")

        # 2. Create an optimized FAISS index (IndexIVFFlat)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)

        # 3. Train the index
        start_time = time.time()
        with st.spinner('Training the FAISS index...'):
            self.index.train(embeddings.astype('float32'))
        st.success(f"Index trained in {time.time() - start_time:.2f} seconds.")

        # 4. Add the embeddings to the index
        self.index.add(embeddings.astype('float32'))
        st.success(f"Successfully added {self.index.ntotal} vectors to the index.")

    def search(self, query_text, k=5, nprobe=2):
        """Searches for similar sentences using a text query."""
        if self.index is None:
            raise RuntimeError("Index has not been built. Please call `build_index` first.")
        query_vector = self.model.encode([query_text])
        return self.search_by_vector(query_vector, k, nprobe)

    def search_by_vector(self, query_vector, k=5, nprobe=2):
        """Searches for similar sentences using a query vector."""
        if self.index is None:
            raise RuntimeError("Index has not been built. Please call `build_index` first.")
        self.index.nprobe = nprobe
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 for no result
                results.append((self.sentences[idx], distances[0][i]))
        return results

    def save_index(self, path="my_faiss_index"):
        """Saves the FAISS index and sentences to a directory."""
        st.write(f"\nSaving index to '{path}'...")
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "sentences.pkl"), "wb") as f:
            pickle.dump(self.sentences, f)
        st.success("Index saved successfully.")

    def load_index(self, path="my_faiss_index"):
        """Loads the FAISS index and sentences from a directory."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "sentences.pkl"), "rb") as f:
            self.sentences = pickle.load(f)
        # Ensure embedding_dim is set after loading
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        st.success(f"Index with {self.index.ntotal} vectors loaded successfully.")

# ==============================================================================
# Streamlit Web App
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🔎 FAISS-Powered Vector Search Engine")

# --- Initialize session state ---
if 'vector_search' not in st.session_state:
    with st.spinner('Loading sentence transformer model...'):
        st.session_state.vector_search = VectorSearch()
    st.success("Model loaded successfully.")
if 'sentences_from_csv' not in st.session_state:
    st.session_state.sentences_from_csv = ""


# --- Sidebar for controls ---
with st.sidebar:
    st.header("Controls")

    # Option to load an existing index
    st.subheader("Load Existing Index")
    index_path_to_load = st.text_input("Index Directory to Load", "my_faiss_index")
    if st.button("Load Index"):
        try:
            st.session_state.vector_search.load_index(index_path_to_load)
            # When loading an index, clear any sentences from a previous CSV upload
            st.session_state.sentences_from_csv = "\n".join(st.session_state.vector_search.sentences)
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    st.divider()

    # Option to build a new index
    st.subheader("Build a New Index")

    # --- NEW: CSV Uploader ---
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # To read file as string:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            
            # Check for 'sentences' column
            if "sentences" in df.columns:
                # Convert column to a list of strings and join them with newlines
                sentences_list = df["sentences"].astype(str).tolist()
                st.session_state.sentences_from_csv = "\n".join(sentences_list)
                st.success(f"Successfully loaded {len(sentences_list)} sentences from CSV.")
            else:
                st.error("Error: The CSV file must contain a column named 'sentences'.")
                st.session_state.sentences_from_csv = "" # Clear previous data on error
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")
            st.session_state.sentences_from_csv = ""


    sentences_input = st.text_area(
        "Enter sentences (one per line) or upload a CSV",
        value=st.session_state.sentences_from_csv, # Use session state to pre-fill
        placeholder="The cat sat on the mat.\nThe dog chased the cat.",
        height=250
    )
    nlist = st.slider("Number of Clusters (nlist)", 1, 10, 2)

    if st.button("Build Index"):
        sentences = [s.strip() for s in sentences_input.split('\n') if s.strip()]
        if sentences:
            st.session_state.vector_search.build_index(sentences, nlist=nlist)
        else:
            st.warning("Please enter at least one sentence or upload a valid CSV.")

    st.divider()

    # Option to save the current index
    st.subheader("Save Current Index")
    index_path_to_save = st.text_input("Index Directory to Save", "my_faiss_index")
    if st.button("Save Index"):
        if st.session_state.vector_search.index is not None:
            st.session_state.vector_search.save_index(index_path_to_save)
        else:
            st.warning("No index has been built or loaded yet.")


# --- Main content area for search ---
st.header("Perform a Search")

if st.session_state.vector_search.index is None:
    st.info("Please build or load an index using the controls in the sidebar.")
else:
    st.success(f"**Index is ready!** It contains {st.session_state.vector_search.index.ntotal} sentences.")

    query_text = st.text_input("Enter your search query:", "a domestic animal")

    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of results (k)", 1, 10, 5)
    with col2:
        # Ensure nprobe doesn't exceed nlist
        max_nprobe = st.session_state.vector_search.index.nlist
        nprobe = st.slider("Clusters to search (nprobe)", 1, max_nprobe, min(1, max_nprobe))


    if st.button("Search"):
        if query_text:
            start_time = time.time()
            results = st.session_state.vector_search.search(query_text, k=k, nprobe=nprobe)
            end_time = time.time()

            st.write(f"Search completed in {end_time - start_time:.4f} seconds.")
            st.subheader("Search Results")

            if results:
                for sentence, distance in results:
                    st.write(f"- **Sentence:** *{sentence}* \n  - **Distance:** `{distance:.4f}`")
            else:
                st.write("No results found.")
        else:
            st.warning("Please enter a search query.")
