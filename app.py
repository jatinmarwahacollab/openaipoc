import openai
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv('openai_api_key')  # Replace with your actual OpenAI API key

# Initialize OpenAI client, passing the API key
client = openai.Client(api_key=openai.api_key)  # Pass the API key here


def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv('pinecone_api_key')  # Use your actual Pinecone API key here
)

index_name = 'pdf-embeddings-index-openai-chunks'
index = pc.Index(index_name)


def get_query_embedding(query):
    # Compute the embedding of the query using the OpenAI model
    embedding = get_openai_embedding(query)
    return embedding

def query_index(query, top_k=2):
    query_embedding = get_query_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True  # Ensure metadata is included in the response
    )
    return results


def generate_gemini_response(prompt, gemini_api_key):
    # Access your API key as an environment variable.
    genai.configure(api_key=gemini_api_key)
    # Choose a model that's appropriate for your use case.
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Add instruction for concise responses in bullet points
    prompt = f"{prompt}\n\nPlease provide a concise answer in bullet points only when answer is longer than 2 lines, also as if the job candidate is directly responding to the recruiter. Always impersonate as a candidate, If you think context is not good then try to answer it using your own general knowledge"
    response = model.generate_content(prompt)
    return response.text

# Replace gemini_api_key with your actual API key for Gemini
gemini_api_key = os.getenv('gemini_api_key')

# Streamlit UI
st.title("Jatin Marwaha's Resume")

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Form for user input
with st.form(key='question_form'):
    user_question = st.text_input("Ask any question about Jatin's professional experience:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if user_question:
        # Query the vector database with the user's question
        results = query_index(user_question)

        # Check if the top match is above a similarity threshold
        threshold = 0.3  # Adjust this threshold based on your needs
        best_match = results['matches'][0] if results['matches'] else None
        if best_match and best_match['score'] >= threshold:
            # Use the best match from the index
            context_snippets = best_match['metadata']['text']
            prompt = f"Based on the following context, answer the question:\n\nContext:\n{context_snippets}\n\nQuestion: {user_question}\nAnswer:"
            gemini_response = generate_gemini_response(prompt, gemini_api_key)
        else:
            # Use previous history and general knowledge
            history_context = "\n".join(st.session_state.history[-3:])  # Keep only the last 3 interactions
            prompt = f"You are a LLM with immense internet knowledge, Based on the previous history try to gather important keywords and then answer the question to the best of your knowledge, if history is not available or it is not making sense, try to answer it anyways with your immense knowledge just mention that you didn't find the relevant information but to the best of my knowledge:\n\nHistory:\n{history_context}\n\nQuestion: {user_question}\nAnswer:"
            gemini_response = generate_gemini_response(prompt, gemini_api_key)

        # Display the response from Gemini
        st.markdown("**Jatin:**")
        st.markdown(gemini_response)

        # Add user question and response to history
        st.session_state.history.append(f"Question: {user_question}")
        st.session_state.history.append(f"Answer: {gemini_response}")