import streamlit as st
from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import os

# API keys
os.environ["OPENAI_API_KEY"] = ""

dataframe = pd.read_csv("D:\LLMs\FProject_2\data\insurance500.csv")

# Documents aligning for FAISS
documents = []
for index, row in dataframe.iterrows():
    documents.append(Document(page_content=row["answer"], metadata={"category": row["category"], "subCategory": row["subCategory"]}))

# Generating embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Top 3 relevant contexts

# Prompt template
template = """Answer the question using only the provided context. Be concise and provide estimates when requested. If the context is insufficient, state that you lack enough information:
{context}

Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Streamlit Interface
st.title("Insurance Chatbot")

# Model selection
model_choice = st.radio("Choose the model to use:", ("GPT", "LLaMA"))

if model_choice == "GPT":
    llm = OpenAI(temperature=0.3, max_tokens=1024)
elif model_choice == "LLaMA":
    llm = Ollama(base_url="http://localhost:11434", model="llama3.1:latest")

# Input query
query = st.text_input("Enter your question:")

if query:
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Generate answer using the chosen model
    gpt_output = llm(prompt.format(context=context, question=query))

    # Display results
    st.subheader("Relevant Context")
    for i, doc in enumerate(retrieved_docs, start=1):
        st.write(f"Context {i}: {doc.page_content}")

    st.subheader("Generated Answer")
    st.write(gpt_output)
