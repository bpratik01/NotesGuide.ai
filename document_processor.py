import os
import tempfile
from langchain.document_loaders import PyPDFLoader, WebBaseLoader

class DocumentProcessor:
    """Class to handle processing of different document types"""
    
    def process_pdfs(self, uploaded_files):
        
        documents = []
        
        for uploaded_file in uploaded_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            
            try:
                # Load PDF using LangChain's loader
                loader = PyPDFLoader(temp_path)
                pdf_documents = loader.load()
                
                # Add source metadata
                for doc in pdf_documents:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = uploaded_file.name
                
                documents.extend(pdf_documents)
            except Exception as e:
                raise Exception(f"Error processing PDF {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        return documents
    
    def process_website(self, url):
        """Process website URL and return documents
        
        Args:
            url: Website URL to process
            
        Returns:
            List of document objects
        """
        try:
            loader = WebBaseLoader(url)
            web_documents = loader.load()
            
            # Add source metadata
            for doc in web_documents:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = url
            
            return web_documents
        except Exception as e:
            raise Exception(f"Error processing website {url}: {str(e)}")