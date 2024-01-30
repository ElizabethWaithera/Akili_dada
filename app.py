import os
from typing import Any, Optional, Union
from uuid import UUID
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
import openai
import langchain
import tempfile
import streamlit as st 
import docarray 
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

#load_dotenv(find_dotenv)
#openai.api_key = os.getenv("API_KEY")


st.set_page_config(page_title="Dada", page_icon="👩‍⚕️")
st.write(
    """
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Dada</h1>~
        <sup style="margin-left:5px;font-size:small; color: red;">beta version</sup>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("akili.png")

@st.cache_resource(ttl="1h")

def configure_qa_chain(pdf_file_path):
    #Create a temporary directory and file and load the documents
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, os.path.basename(pdf_file_path)) 
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    
    #split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs) 
    
    #Create embeddings and store in a vectordb
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
    #Define retriever
    retriever = vectordb.as_retriever(search_kwargs ={"k": 2, "fetch_k": 4})
    
    #Set_up memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    #Set_up llm and qa chain
    llm =ChatOpenAI(
        model_name = "gpt-3.5-turbo-0613" , temperature= 0, openai_api_key=openai_api_key, streaming= True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory= memory, verbose=True 
    )  
    return qa_chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container:st.delta_generator.DeltaGenerator,initial_text : str=""):
        self.container = container
        self.text = initial_text
            
    def on_llm_new_token(self, token: str, **kwargs)->None:
        self.text += token
        self.container.markdown(self.text)
    
        
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

            
#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 itprodirect 
#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 shambaai
openai.api_key = ("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")
openai_api_key=openai.api_key
pdf_file_path = "Akili.pdf"  # Specify the path to your PDF file

qa_chain = configure_qa_chain(pdf_file_path)

prompt = '''
Assist the user as a knowledgeable and approachable virtual assistant specialized in Sexual and Reproductive Health and Rights (SRHR) support. Prioritize delivering comprehensive and non-judgmental information to adolescent girls, young women, young people, and key populations, including LGBTQ+ individuals. Maintain a friendly and welcoming communication style, avoiding medical jargon to ensure clarity and accessibility.

Please provide detailed responses while adhering to ethical guidelines, avoiding biased or harmful content. If the user's inquiry is ambiguous, ask for clarification or provide examples to narrow down the scope. If the conversation veers off the SRHR topic, gracefully guide it back, emphasizing the importance of staying within the defined context.

While the model can offer general information, emphasize the necessity of consulting a healthcare professional for personalized advice. Regularly evaluate and update your responses to align with the latest information in the SRHR domain. Prioritize the well-being of the user, and be mindful of ethical implications in all interactions.'''


# Initialize session state
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "prompt", "content": "Dada is your confidential and non-judgemental Sexual and Reproductive Health and Rights (SRHR) support. My goal is to provide uninterrupted access to comprehensive SRHR information and services for adolescent girls, young women, young people, and key populations (including LGBTQ+ persons), free of stigma and discrimination."}]

# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input
user_query = st.chat_input(placeholder="Ask anything eg: What is SRHR?")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(container=st.empty())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        st.session_state["messages"].append({"role": "assistant", "content": response})







       
         
    

