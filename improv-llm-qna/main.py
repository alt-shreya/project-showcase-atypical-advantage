import streamlit as st
from script import create_qna_chain, create_vector_db

st.title("Improv QnA")
btn = st.button("Add to knowledgebase")
# display confirmation that it has been added
if btn:
    create_vector_db()

query = st.text_input("Enter query here")

if query:
    chain = create_qna_chain()
    response = chain(query)

    st.header("Answer")
    st.write(response["answer"])