from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
load_dotenv() 

class VectorStoreManager:
    """Class to handle creation and management of vector stores"""
    
    def __init__(self, embedding_model=None):
       
        # Use provided embedding model or default to OpenAI with Streamlit secrets
        self.embeddings = embedding_model or OpenAIEmbeddings(
            openai_api_key=st.secrets.get("OPENAI_API_KEY", None)
        )
        
        # Configure text splitter with default parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000,
            chunk_overlap=500
        )
    
    def create_vector_store(self, documents):
        """Create a FAISS vector store from documents
        
        Args:
            documents: List of document objects
            
        Returns:
            Tuple of (list of text chunks, FAISS vector store)
        """
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        return chunks, vector_store
    
    def save_vector_store(self, vector_store, path):
        """Save the vector store to disk
        
        Args:
            vector_store: FAISS vector store
            path: Path to save the vector store
        """
        vector_store.save_local(path)
    
    def load_vector_store(self, path):
        """Load a vector store from disk
        
        Args:
            path: Path to load the vector store from
            
        Returns:
            FAISS vector store
        """
        return FAISS.load_local(path, self.embeddings)