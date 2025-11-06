"""
Research Copilot - Streamlit Frontend App

Interactive web interface for querying research papers using RAG pipeline.
Provides search functionality with citation display and answer formatting.
"""

import streamlit as st
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import QA modules
try:
    from qa.rag import create_rag_pipeline, RAGResponse
    from qa.query_rewriter import create_academic_query_rewriter
    from qa.formatter import AnswerFormatter
except ImportError as e:
    st.error(f"Failed to import QA modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .search-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .citation-box {
        padding: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }
    .confidence-score {
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem 0;
    }
    .confidence-high { background-color: #28a745; }
    .confidence-medium { background-color: #ffc107; color: black; }
    .confidence-low { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class ResearchCopilotApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.initialize_session_state()
        self.load_configuration()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        
        if 'query_rewriter' not in st.session_state:
            st.session_state.query_rewriter = None
        
        if 'formatter' not in st.session_state:
            st.session_state.formatter = AnswerFormatter()
    
    def load_configuration(self):
        """Load configuration from sidebar."""
        st.sidebar.title("🔬 Research Copilot")
        st.sidebar.markdown("---")
        
        # Model configuration
        st.sidebar.subheader("🤖 Model Configuration")
        
        llm_model = st.sidebar.selectbox(
            "LLM Model",
            ["phi4-mini:3.8b", "deepseek-coder-v2", "llama2", "mistral", "codellama"],
            index=0
        )
        
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1"],
            index=0
        )
        
        # Retrieval configuration
        st.sidebar.subheader("🔍 Retrieval Settings")
        
        max_chunks = st.sidebar.slider(
            "Max Chunks to Retrieve",
            min_value=3,
            max_value=20,
            value=5,
            help="Number of document chunks to retrieve for each query"
        )
        
        use_hybrid_search = st.sidebar.checkbox(
            "Use Hybrid Search (FAISS + BM25)",
            value=True,
            help="Combine semantic and keyword search"
        )
        
        use_query_expansion = st.sidebar.checkbox(
            "Enable Query Expansion",
            value=True,
            help="Expand queries with related terms"
        )
        
        # Citation style
        citation_style = st.sidebar.selectbox(
            "Citation Style",
            ["academic", "brief", "full"],
            index=0
        )
        
        # Data paths
        st.sidebar.subheader("📁 Data Paths")
        
        faiss_path = st.sidebar.text_input(
            "FAISS Index Path",
            value="data/processed/faiss_index.bin"
        )
        
        bm25_path = st.sidebar.text_input(
            "BM25 Index Path",
            value="data/processed/bm25_index.pkl"
        )
        
        chunks_db_path = st.sidebar.text_input(
            "Chunks Database Path",
            value="data/processed/chunks.db"
        )
        
        papers_db_path = st.sidebar.text_input(
            "Papers Database Path",
            value="data/processed/papers.db"
        )
        
        # Store configuration in session state
        st.session_state.config = {
            'llm_model': llm_model,
            'embedding_model': embedding_model,
            'max_chunks': max_chunks,
            'use_hybrid_search': use_hybrid_search,
            'use_query_expansion': use_query_expansion,
            'citation_style': citation_style,
            'faiss_path': faiss_path,
            'bm25_path': bm25_path,
            'chunks_db_path': chunks_db_path,
            'papers_db_path': papers_db_path
        }
    
    def initialize_pipeline(self):
        """Initialize RAG pipeline with current configuration."""
        config = st.session_state.config
        
        try:
            with st.spinner("Initializing RAG pipeline..."):
                # Create RAG pipeline
                st.session_state.rag_pipeline = create_rag_pipeline(
                    faiss_index_path=config['faiss_path'],
                    bm25_index_path=config['bm25_path'],
                    chunk_metadata_path=config['chunks_db_path'],
                    papers_db_path=config['papers_db_path'],
                    llm_model=config['llm_model']
                )
                
                # Create query rewriter if enabled
                if config['use_query_expansion']:
                    st.session_state.query_rewriter = create_academic_query_rewriter(
                        ollama_model=config['llm_model']
                    )
                
                # Update formatter
                st.session_state.formatter = AnswerFormatter(
                    citation_style=config['citation_style']
                )
                
                st.success("✅ Pipeline initialized successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize pipeline: {str(e)}")
            return False
    
    def render_search_interface(self):
        """Render the main search interface."""
        st.title("🔬 Research Copilot")
        st.markdown("Ask questions about research papers and get AI-powered answers with citations.")
        
        # Search box
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter your research question:",
                placeholder="e.g., What are the main advantages of transformer architectures?",
                key="search_query"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                custom_max_chunks = st.number_input(
                    "Max Chunks",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.config['max_chunks']
                )
            
            with col2:
                force_reindex = st.checkbox("Force Re-indexing")
            
            with col3:
                output_format = st.selectbox(
                    "Output Format",
                    ["Markdown", "JSON", "HTML"]
                )
        
        return query, search_button, custom_max_chunks, output_format
    
    def process_query(self, query: str, max_chunks: int = 5):
        """Process a user query through the RAG pipeline."""
        if not st.session_state.rag_pipeline:
            st.error("❌ Pipeline not initialized. Please check configuration and try again.")
            return None
        
        try:
            with st.spinner("Processing your query..."):
                # Add progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Query expansion (if enabled)
                status_text.text("Expanding query...")
                progress_bar.progress(20)
                
                expanded_query = query
                if (st.session_state.config['use_query_expansion'] and 
                    st.session_state.query_rewriter):
                    expanded_query = st.session_state.query_rewriter.rewrite(query)
                
                # Step 2: Retrieve documents
                status_text.text("Retrieving relevant documents...")
                progress_bar.progress(50)
                
                # Step 3: Generate answer
                status_text.text("Generating answer...")
                progress_bar.progress(80)
                
                response = st.session_state.rag_pipeline.query(
                    question=query,
                    max_chunks=max_chunks,
                    use_query_rewriter=st.session_state.config['use_query_expansion']
                )
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                # Clear progress indicators
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                return response
                
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            return None
    
    def render_response(self, response: RAGResponse, output_format: str = "Markdown"):
        """Render the RAG response in the specified format."""
        if not response:
            return
        
        # Add to query history
        st.session_state.query_history.append({
            'timestamp': response.timestamp,
            'query': response.query,
            'answer': response.answer,
            'citations': response.citations,
            'confidence': response.confidence
        })
        
        # Display response
        st.markdown("---")
        
        # Header with confidence score
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("📋 Answer")
        
        with col2:
            confidence_class = (
                "confidence-high" if response.confidence >= 0.7 else
                "confidence-medium" if response.confidence >= 0.4 else
                "confidence-low"
            )
            st.markdown(
                f'<div class="confidence-score {confidence_class}">Confidence: {response.confidence:.1%}</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.metric("Processing Time", f"{response.processing_time:.1f}s")
        
        # Display answer based on format
        if output_format == "JSON":
            json_response = st.session_state.formatter.format_as_json(
                response.answer,
                response.citations,
                response.retrieved_chunks,
                response.confidence,
                response.query,
                response.processing_time
            )
            st.code(json_response, language="json")
        
        elif output_format == "HTML":
            html_response = st.session_state.formatter.format_as_html(
                response.answer,
                response.citations,
                response.retrieved_chunks,
                response.confidence,
                response.query
            )
            st.markdown(html_response, unsafe_allow_html=True)
        
        else:  # Markdown (default)
            markdown_response = st.session_state.formatter.format_as_markdown(
                response.answer,
                response.citations,
                response.retrieved_chunks,
                response.confidence,
                response.query
            )
            st.markdown(markdown_response)
        
        # Citations section
        if response.citations:
            st.subheader("📚 Citations")
            
            for i, citation in enumerate(response.citations, 1):
                # Find corresponding chunk for metadata
                chunk_metadata = None
                for chunk in response.retrieved_chunks:
                    if citation in [chunk.paper_id, f"paper_{chunk.paper_id}", 
                                  chunk.metadata.get('arxiv_id', ''), 
                                  chunk.metadata.get('doi', '')]:
                        chunk_metadata = chunk.metadata
                        break
                
                # Display citation
                with st.container():
                    st.markdown(f"**[{i}]** {citation}")
                    if chunk_metadata:
                        if 'title' in chunk_metadata:
                            st.markdown(f"*{chunk_metadata['title']}*")
                        if 'authors' in chunk_metadata:
                            st.markdown(f"Authors: {chunk_metadata['authors']}")
        
        # Retrieved chunks (collapsible)
        with st.expander(f"🔍 Retrieved Chunks ({len(response.retrieved_chunks)})"):
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                st.markdown(f"**Chunk {i}** (Score: {chunk.score:.3f}, Source: {chunk.source})")
                st.markdown(f"Paper: {chunk.paper_id}")
                st.markdown(f"Section: {chunk.metadata.get('section', 'Unknown')}")
                st.text_area(
                    f"Content {i}",
                    chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                    height=100,
                    key=f"chunk_content_{i}"
                )
    
    def render_sidebar_history(self):
        """Render query history in sidebar."""
        if st.session_state.query_history:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📝 Query History")
            
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.sidebar.expander(f"Query {len(st.session_state.query_history) - i}"):
                    st.write(f"**Q:** {item['query'][:100]}...")
                    st.write(f"**Confidence:** {item['confidence']:.1%}")
                    st.write(f"**Time:** {item['timestamp'][:19]}")
    
    def render_pipeline_status(self):
        """Render pipeline status information."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Pipeline Status")
        
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_statistics()
            
            st.sidebar.metric("LLM Model", stats.get('llm_model', 'Unknown'))
            st.sidebar.metric("Total Chunks", stats['retriever_stats']['total_chunks'])
            
            # Status indicators
            faiss_status = "✅" if stats['retriever_stats']['faiss_available'] else "❌"
            bm25_status = "✅" if stats['retriever_stats']['bm25_available'] else "❌"
            
            st.sidebar.write(f"FAISS Index: {faiss_status}")
            st.sidebar.write(f"BM25 Index: {bm25_status}")
            
        else:
            st.sidebar.write("❌ Pipeline not initialized")
            if st.sidebar.button("Initialize Pipeline"):
                self.initialize_pipeline()
    
    def run(self):
        """Run the main application."""
        # Render sidebar components
        self.render_sidebar_history()
        self.render_pipeline_status()
        
        # Initialize pipeline if not already done
        if not st.session_state.rag_pipeline:
            st.info("🔧 Pipeline not initialized. Click 'Initialize Pipeline' in the sidebar to get started.")
            return
        
        # Render main interface
        query, search_button, max_chunks, output_format = self.render_search_interface()
        
        # Process query on button click or Enter
        if (search_button or query) and query.strip():
            response = self.process_query(query, max_chunks)
            if response:
                self.render_response(response, output_format)
        
        # Sample queries section
        st.markdown("---")
        st.subheader("💡 Sample Queries")
        
        sample_queries = [
            "What are the main advantages of transformer architectures?",
            "How does attention mechanism work in neural networks?",
            "Compare CNNs and transformers for computer vision tasks",
            "What are the latest developments in large language models?",
            "Explain the concept of few-shot learning in machine learning"
        ]
        
        cols = st.columns(len(sample_queries))
        for i, sample_query in enumerate(sample_queries):
            with cols[i]:
                if st.button(f"Try: {sample_query[:30]}...", key=f"sample_{i}"):
                    st.session_state.search_query = sample_query
                    st.experimental_rerun()

# Main application entry point
def main():
    """Main function to run the Streamlit app."""
    app = ResearchCopilotApp()
    app.run()

if __name__ == "__main__":
    main()
