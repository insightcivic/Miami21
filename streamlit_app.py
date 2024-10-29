import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Load OpenAI API key from Streamlit Secrets
openai.api_key = st.secrets["openai_api_key"]

# Load the FAISS index and the chunked text data
index = faiss.read_index("miami21_index.faiss")
with open("miami21_chunked_with_overlap.txt", "r", encoding="utf-8") as file:
    chunked_sections_with_overlap = file.read().split("\n\n")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')


# Function to retrieve relevant chunks
def retrieve_chunks(query, top_k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = [chunked_sections_with_overlap[i] for i in indices[0]]
    return retrieved_chunks

# Function to generate response using OpenAI GPT-4
def generate_response(query, retrieved_chunks):
    # Combine retrieved chunks into a single context string
    context = "\n\n".join(retrieved_chunks)
    
    # Create a structured message for the chat completion
    messages = [
        {"role": "system", "content": "You are an assistant providing information based on the Miami 21 code."
                                      "Use the provided context to answer the user's question as hlepfully and accurately as possible."
                                      "If the context does not contain the answer on a specific question, respond with, "
                                       "'I'm only able to answer based on the information in the Miami 21 Code.' "
                                       },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    # Check the latest API documentation for the correct method to use; typically,
    # the ChatCompletion might have been renamed or modified. In the meantime, if 
    # this doesn't work, refer to the OpenAI migration tool or guides.
    try:
        response = openai.chat.completions.create(  # Tentative; update based on latest documentation
            model="gpt-4o-mini",  # Specify the latest model or correct endpoint
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )
        #response = openai.completions.chat(  # Tentative; update based on latest documentation
        #    model="gpt-4o mini",  # Specify the latest model or correct endpoint
        #    messages=messages,
        #    max_tokens=150,
        #    temperature=0.5
        #)
        
        # Extract the assistant's reply from the response
        return response.choices[0].message.content.strip()
    
    except AttributeError as e:
        # Print the error for debugging; remove this in production
        print(f"AttributeError: {e}")
        st.error("Error with OpenAI API call. Please check API version compatibility.")



# Streamlit UI
st.title("Ask Miami 21")
st.write("Enter a question related to the Miami 21 code, and the system will retrieve relevant sections and generate an answer.")

query = st.text_input("Your Question:")
if st.button("Get Answer") and query:
    with st.spinner("Retrieving relevant information and generating answer..."):
        retrieved_chunks = retrieve_chunks(query)
        answer = generate_response(query, retrieved_chunks)
        st.subheader("Answer:")
        st.write(answer)
        
        st.subheader("Relevant Sections:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.write(f"**Section {i}:**\n{chunk}")
