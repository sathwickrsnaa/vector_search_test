import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os
import pickle
import streamlit as st
import pandas as pd
import io
import networkx as nx



class VectorSearch:
    """
    A class for performing optimized similarity search and clustering on sentence embeddings.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.sentences = []
        self.embeddings = None

    def build_index(self, sentences, nlist=2):
        """Builds an optimized FAISS index from a list of sentences."""
        st.write(f"\nBuilding index for {len(sentences)} sentences...")
        self.sentences = sentences

        start_time = time.time()
        
        st.write("Creating embeddings...")
        progress_bar = st.progress(0)
        all_embeddings = []
        batch_size = 32 # Process sentences in batches
        for i in range(0, len(self.sentences), batch_size):
            batch = self.sentences[i:i+batch_size]
            embeddings_batch = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(embeddings_batch)
            progress_percentage = min((i + batch_size) / len(self.sentences), 1.0)
            progress_bar.progress(progress_percentage)

        self.embeddings = np.vstack(all_embeddings).astype('float32')
        progress_bar.empty() # Remove the progress bar after completion
        st.success(f"Embeddings created in {time.time() - start_time:.2f} seconds.")

        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)

        start_time = time.time()
        with st.spinner('Training the FAISS index...'):
            self.index.train(self.embeddings)
        st.success(f"Index trained in {time.time() - start_time:.2f} seconds.")

        self.index.add(self.embeddings)
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
            if idx != -1:
                results.append((self.sentences[idx], distances[0][i]))
        return results

    def find_similarity_clusters(self, threshold, k=10, nprobe=2):
        """
        Groups sentences into clusters based on a similarity threshold.
        """
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Index has not been built. Please call `build_index` first.")

        self.index.nprobe = nprobe
        distances, indices = self.index.search(self.embeddings, k)

        G = nx.Graph()
        all_sentences = list(self.sentences)
        G.add_nodes_from(all_sentences)

        for i in range(len(all_sentences)):
            for j_idx, dist in zip(indices[i], distances[i]):
                if i != j_idx and dist < threshold:
                    G.add_edge(all_sentences[i], all_sentences[j_idx])

        clusters = list(nx.connected_components(G))
        
        sentence_to_cluster_id = {}
        multi_member_clusters = []
        cluster_id_counter = 0
        for component in clusters:
            if len(component) > 1:
                multi_member_clusters.append(list(component))
                for sentence in component:
                    sentence_to_cluster_id[sentence] = cluster_id_counter
                cluster_id_counter += 1
        
        for sentence in all_sentences:
            if sentence not in sentence_to_cluster_id:
                sentence_to_cluster_id[sentence] = -1
        
        export_df = pd.DataFrame(sentence_to_cluster_id.items(), columns=['sentence', 'cluster_id'])
        
        return multi_member_clusters, export_df

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
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        st.write("Preparing embeddings for loaded index...")
        progress_bar = st.progress(0)
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(self.sentences), batch_size):
            batch = self.sentences[i:i+batch_size]
            embeddings_batch = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(embeddings_batch)
            progress_percentage = min((i + batch_size) / len(self.sentences), 1.0)
            progress_bar.progress(progress_percentage)
            
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        progress_bar.empty()
        st.success(f"Index with {self.index.ntotal} vectors loaded successfully.")

# ==============================================================================
# Streamlit Web App (Updated)
# ==============================================================================

st.set_page_config(layout="wide")

# --- NEW: Title, Caption, and Description ---
st.title("🚀 Newtelligence")
st.caption("Search, Connect, Discover")
st.markdown("""
Welcome to **Newtelligence**, your intelligent tool for text analysis. This application leverages the power of sentence embeddings and high-speed vector search to unlock insights from your text data.

**What you can do:**
- **Build or Load an Index:** Convert your sentences into a searchable, high-performance index.
- **Semantic Search:** Find sentences that are most similar to your query, not just by keywords, but by meaning.
- **Discover Clusters:** Automatically group related sentences into thematic clusters based on a similarity threshold you control.
""")


# --- Initialize session state ---
if 'vector_search' not in st.session_state:
    with st.spinner('Loading sentence transformer model...'):
        st.session_state.vector_search = VectorSearch()
    st.success("Model loaded successfully.")
if 'sentences_from_csv' not in st.session_state:
    st.session_state.sentences_from_csv = ""
if 'cluster_export_data' not in st.session_state:
    st.session_state.cluster_export_data = None


# --- Sidebar for controls ---
with st.sidebar:
    st.header("Controls")

    st.subheader("Load Existing Index")
    index_path_to_load = st.text_input("Index Directory to Load", "my_faiss_index")
    if st.button("Load Index"):
        try:
            st.session_state.vector_search.load_index(index_path_to_load)
            st.session_state.sentences_from_csv = "\n".join(st.session_state.vector_search.sentences)
            st.session_state.cluster_export_data = None # Reset export data
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    st.divider()

    st.subheader("Build a New Index")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            if "sentences" in df.columns:
                sentences_list = df["sentences"].astype(str).tolist()
                st.session_state.sentences_from_csv = "\n".join(sentences_list)
                st.success(f"Successfully loaded {len(sentences_list)} sentences from CSV.")
            else:
                st.error("Error: The CSV file must contain a column named 'sentences'.")
                st.session_state.sentences_from_csv = ""
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")
            st.session_state.sentences_from_csv = ""

    sentences_input = st.text_area(
        "Enter sentences (one per line) or upload a CSV",
        value=st.session_state.sentences_from_csv,
        placeholder="The cat sat on the mat.\nThe dog chased the cat.",
        height=250
    )
    nlist = st.slider("Number of Clusters (nlist)", 1, 10, 2)

    if st.button("Build Index"):
        sentences = [s.strip() for s in sentences_input.split('\n') if s.strip()]
        if sentences:
            st.session_state.vector_search.build_index(sentences, nlist=nlist)
            st.session_state.cluster_export_data = None # Reset export data
        else:
            st.warning("Please enter at least one sentence or upload a valid CSV.")

    st.divider()

    st.subheader("Save Current Index")
    index_path_to_save = st.text_input("Index Directory to Save", "my_faiss_index")
    if st.button("Save Index"):
        if st.session_state.vector_search.index is not None:
            st.session_state.vector_search.save_index(index_path_to_save)
        else:
            st.warning("No index has been built or loaded yet.")

# --- Main content area ---
if st.session_state.vector_search.index is None:
    st.info("Please build or load an index using the controls in the sidebar.")
else:
    st.success(f"**Index is ready!** It contains {st.session_state.vector_search.index.ntotal} sentences.")

    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Perform a Search")
        query_text = st.text_input("Enter your search query:", "a domestic animal")
        k_search = st.slider("Number of results (k)", 1, 10, 5, key="k_search")
        max_nprobe = st.session_state.vector_search.index.nlist
        nprobe_search = st.slider("Clusters to search (nprobe)", 1, max_nprobe, min(1, max_nprobe), key="nprobe_search")

        if st.button("Search"):
            if query_text:
                start_time = time.time()
                results = st.session_state.vector_search.search(query_text, k=k_search, nprobe=nprobe_search)
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

    with col2:
        st.header("2. Find Similarity Clusters")
        help_text = "L2 distance. A smaller value (e.g., < 0.5) means higher similarity. A value too high may return too many results."
        similarity_threshold = st.number_input(
            "Similarity Threshold",
            min_value=0.0, max_value=2.0, value=0.75, step=0.05, help=help_text
        )
        
        if st.button("Find Clusters"):
            with st.spinner("Finding clusters... This might take a moment."):
                start_time = time.time()
                clusters, export_df = st.session_state.vector_search.find_similarity_clusters(threshold=similarity_threshold)
                end_time = time.time()
                st.session_state.cluster_export_data = export_df.to_csv(index=False).encode('utf-8')

            st.write(f"Clustering completed in {end_time - start_time:.4f} seconds.")
            
            # --- NEW: Logic to handle > 500 clusters and display unassigned ---
            clusters.sort(key=len, reverse=True) # Sort by size
            
            st.subheader(f"Found {len(clusters)} Clusters")

            clusters_to_display = clusters
            if len(clusters) > 500:
                st.info(f"Displaying the top 100 largest clusters out of {len(clusters)} found.")
                clusters_to_display = clusters[:100]

            if clusters_to_display:
                for i, cluster in enumerate(clusters_to_display):
                    with st.expander(f"Cluster {i} ({len(cluster)} sentences)"):
                        for sentence in cluster:
                            st.markdown(f"- *{sentence}*")
            else:
                st.info("No clusters found with more than one member. Try increasing the threshold.")
            
            # Display unassigned sentences
            unassigned_sentences = export_df[export_df['cluster_id'] == -1]['sentence'].tolist()
            if unassigned_sentences:
                with st.expander(f"Unassigned Sentences ({len(unassigned_sentences)} sentences)"):
                    for sentence in unassigned_sentences:
                        st.markdown(f"- *{sentence}*")


        if st.session_state.cluster_export_data is not None:
            st.download_button(
               label="Download Clusters as CSV",
               data=st.session_state.cluster_export_data,
               file_name='sentence_clusters.csv',
               mime='text/csv',
            )
