import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from question_answerer import QuestionAnswerer

# Set page configuration
st.set_page_config(page_title="NotesGuide.ai", layout="wide")

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

class RAGApplication:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.question_answerer = QuestionAnswerer(model="llama-3.3-70b-versatile", temperature=0.3)
        
    def render_ui(self):
        """Render the main Streamlit UI components"""
        st.title("NotesGuide.ai")
        st.write("Upload your study materials and ask questions to get relevant answers.")
        
        self.render_sidebar()
        
        # Main area for questions and answers
        self.render_main_area()
    
    def render_sidebar(self):
        """Render the sidebar with upload options"""
        with st.sidebar:
            st.header("Upload Study Materials")
            uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])
            website_url = st.text_input("Or enter a website URL")
            process_button = st.button("Process Materials")
            
            if process_button:
                self.process_materials(uploaded_files, website_url)
    
    def process_materials(self, uploaded_files, website_url):
        """Process uploaded files and websites"""
        documents = []
        
        # Process PDFs
        if uploaded_files:
            with st.spinner("Processing PDF files..."):
                pdf_documents = self.doc_processor.process_pdfs(uploaded_files)
                documents.extend(pdf_documents)
        
        # Process website if URL provided
        if website_url:
            with st.spinner(f"Processing {website_url}..."):
                try:
                    web_documents = self.doc_processor.process_website(website_url)
                    documents.extend(web_documents)
                except Exception as e:
                    st.error(f"Failed to load website content: {e}")
        
        if documents:
            # Create vector store from documents
            with st.spinner("Creating vector embeddings... This might take a minute."):
                chunks, vector_store = self.vector_store_manager.create_vector_store(documents)
                
                st.session_state.vector_store = vector_store
                st.session_state.documents = documents
                st.session_state.processed = True
            
            st.sidebar.success(f"Processed {len(documents)} documents into {len(chunks)} chunks!")
        else:
            st.sidebar.warning("Please upload a PDF or enter a valid URL.")
    
    def render_main_area(self):
        """Render the main area for questions and answers"""
        st.header("Ask Questions About Your Materials")
        user_question = st.text_input("Enter your question:")
        
        if st.button("Get Answer") and user_question:
            if not st.session_state.processed:
                st.warning("Please upload and process study materials first.")
            else:
                with st.spinner("Searching through your materials..."):
                    # Get answer from the question answerer
                    docs, answer = self.question_answerer.answer_question(
                        user_question, 
                        st.session_state.vector_store
                    )
                    
                    # Display results
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    # Show sources
                    st.subheader("Source Material:")
                    for i, doc in enumerate(docs):
                        with st.expander(f"Source {i+1}"):
                            st.write(doc.page_content)
                            if hasattr(doc.metadata, 'source') and doc.metadata.source:
                                st.write(f"Source: {doc.metadata.source}")

# Run the application
if __name__ == "__main__":
    app = RAGApplication()
    app.render_ui()