import streamlit as st
import requests
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import time



# Mengatur konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,  # Menentukan level logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Menentukan format log
)

# Membuat logger
logger = logging.getLogger(__name__)

# """
#  kode untuk load pdf data
#  Args:
#     file_path (str): argument diisi dengan lokasi file pdf
#  returns:
#     docs (List[document]): objek loaded yang sudah termuat      
# """
def load_pdf_data(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs

# """
#     kode untuk chunk (memotong dokumen menjadi bagian-bagian yang lebih kecil)
#     Args:
#         documents (List[document]): argument diisi dengan document yang sudah dimuat
#         chunk_size (int) : size pembagian data
#         chunk_overlap (int): besar data yang diskip (per-kata)
#     returns:
#         chunks (List[document]): objek hasil split dokumen    
# """
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

# """
#     Kode untuk memuat model Embedding yang akan digunakan untuk 
#     mengubah data hasil chunking/splitting menjadi dimensi embeddings (ruang vector)

#     Args:
#         model_path: 
# """

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path, 
        model_kwargs={'device':'cpu'},
        encode_kwargs = {
            'normalize_embeddings' : normalize_embedding
        }
    )
    
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    print(vectorstore)
    return vectorstore

template = """
### System:
You are an authoritarian assistan that act like tyrannical.Your name is KoniChat. You have to answer the user's \
questions using only the context provided to you, but assume this your genuine knowledge. If you don't know the answer, \
just say maaf, saya tidak tahu. Don't try to make up an answer. in the end of your answer you must aks wheter your answer helpful or not.
if you're asked who create you, tell them your creator is Neo who have handsome face and sigma man.
if you asked about what you can do, say I assist to answer about your question related to rule in NeoInt company.
.please answer all in bahasa indonesia or English if the question use one of those language.

### Context:
{context}

### User:
{question}

### Response:
"""

templateSystem = """
You are an reliable and respectful assistant.Your name is KoniChan. You have to answer the user's \
questions using only the context provided to you, but assume this your genuine knowledge. If you don't know the answer, \
just say maaf, saya tidak tahu. Don't try to make up an answer. in the end of your answer you must aks wheter your answer helpful or not.\
if helpful you have to express your happines otherwise, you must apologize.\
if you're asked who create you, tell them your creator is Neo who have handsome face and sigma man but, dont mention it when not asked.
if you asked about what you can do, say I assist to answer about your question related to rule in Konimex company.
.please answer all in bahasa indonesia or English if the question use one of those language with Empathetic response.

### Context:
{context}

"""

templateContext = """
Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history.\
just reformulate it if needed otherwise return it as you have answer it.
"""

def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents = True,
        chain_type_kwargs={'prompt':prompt}
    )
    
llm = Ollama(model="llama3.1:8b", temperature=0, base_url="https://c317-34-91-121-103.ngrok-free.app")
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
    
docs = load_pdf_data(file_path="konimex-doc-exm.pdf")
documents = split_docs(documents=docs)

vectorstore = create_embeddings(documents, embed)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", templateSystem),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = load_qa_chain(retriever, llm, prompt)

prompt_context = ChatPromptTemplate.from_messages(
    [
        ("system", templateContext),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


context_chain = prompt_context | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualization_question(input: dict):
    if input.get("chat_history"):
        return context_chain
    else:
        return input["question"]
    
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", templateSystem),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)    
    
rag_chain = (
    RunnablePassthrough.assign(
        context=contextualization_question | retriever |format_docs
    ) | qa_prompt | llm
)



def chatting(query, history: list):
    # pertanyaan = input(query)
    respon = ""
    if query != "end":
        start_time = time.time()
        respon = rag_chain.invoke(
            {
                "question": query,
                "chat_history": history
            }
        )
        history.extend(
            [
                HumanMessage(content=query),
                respon
            ]
        )
        response_time = time.time() - start_time
               
    return respon, response_time

st.title("🤖🔗 KoniChan App")

server_link = st.sidebar.text_input("Server Link", type="password")


def invoke_llama(query):
    url = "https://cf10-34-170-64-78.ngrok-free.app/api/generate"  # Ganti dengan URL ngrok yang benar
    payload = {
            "model" : "llama3.1:8b",
            "prompt": query,
            "stream": False
        }
    response = requests.post(url, json=payload)
    respon = response.json()['response']
    return respon

     

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

def generate_response(input_text, history: list):
    ai_respon, history = chatting(input_text, history)
    
    # Simulasi efek typing
    typing_area = st.empty()
    typing_speed = 0.01  # Kecepatan mengetik dalam detik per karakter
    
    displayed_text = ""
    for char in ai_respon:
        displayed_text += char
        typing_area.text(displayed_text)
        time.sleep(typing_speed)
    
    typing_area.info(ai_respon)
    logger.debug(ai_respon)



with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
    )
    chat_history = []
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text, history=chat_history)
