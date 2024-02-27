from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from langchain_core.prompts import PromptTemplate




from dotenv import load_dotenv
# load API key from .env file
load_dotenv()
import os
GOOGLE_API_KEY  = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    google_api_key = GOOGLE_API_KEY, 
    temperature = 0.9, 
    convert_system_message_to_human=True)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key = GOOGLE_API_KEY)


vector_filepath = "faiss_db"
qna_filepath = "improv-qna.csv"
user_query = "do you have workshops for beginners?"
# create vector database
def create_vector_db():
    # load qna
    loader = CSVLoader(
        file_path=qna_filepath, 
        source_column="Questions")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    db = FAISS.from_documents(
        documents=data,
        embedding=embeddings)

    # Save vector database locally
    # swap this for server file path in production
    db.save_local(vector_filepath)

# create conversational chain
def create_qna_chain(llm=llm):
    # load local copy of vector db
    db = FAISS.load_local(vector_filepath, embeddings)
    retriever = db.as_retriever(score_threshold=0.8)

    prompt_template = """Given the following context and a question, generate an answer based on this context.
    In the answer try to provide as much relevant text as possible from "Answer" column in the
    source document context, changing what is necessary to fit the question.
    If the answer is not directly found within the context, generate an answer based on the closest matching answer in source_documents.
    You can find this information from the retriever.
    If the question does not have any matches, tell the user,
    "I do not have the information right now, but I can connect you with someone who does! You can reach out to a human here: <human>".

    CONTEXT: {context}
    QUESTION: {question} """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"])
    
    question_generator_chain = LLMChain(llm=llm, prompt=PROMPT)
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=question_generator_chain,
        document_prompt=PROMPT,
        document_variable_name="context"
    )
    # chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever,
        question_generator=question_generator_chain,
)
    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = create_qna_chain(llm=llm)
    print(chain(user_query))