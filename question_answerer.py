from groq import Groq
import streamlit as st

class QuestionAnswerer:
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0.3):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.model = model
        self.temperature = temperature
    
    def answer_question(self, question, vector_store, k=3):
        docs = vector_store.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self._format_prompt(question, context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant that provides comprehensive, detailed responses in a structured notes format."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        answer = response.choices[0].message.content
        return docs, answer
    
    def _format_prompt(self, question, context):
        return f"""
        You are a helpful educational assistant. Use the following context from the student's study materials to answer their question.
        If the context doesn't contain the information needed, say you don't have enough information rather than making up an answer.
        
        Important instructions for formatting your response:
        1. Provide detailed, comprehensive information organized as study notes
        2. Use clear section headings and subheadings with ## and ### markdown formatting
        3. Include numbered or bulleted lists for key points
        4. Highlight important terms, definitions, or concepts in **bold**
        5. Provide examples where appropriate
        6. Structure your answer with a clear introduction, main content sections, and a summary
        7. Include any relevant formulas, procedures, or methodologies
        8. Make connections between concepts when possible
        9. Explain complex ideas step-by-step with clear reasoning
        
        Context:
        {context}
        
        Question: {question}
        """