"""
    **BookPal-AI** is a Python application designed to enhance your reading experience with the power of Language Models (LLMs).
    You can engage in natural language conversations about your documents, which may include PDFs, CSVs, or plain text files.
    BookPal-AI utilizes advanced language models, such as ChatOpenAI or instructor-xl, to provide accurate responses based on the content of your uploaded documents.
    Keep in mind that the application will respond only to questions related to the loaded documents.

    ### How to Use:
    1. **Select an LLM Model:**
        - Choose between "ChatOpenAI" and "instructor-xl" in the sidebar.
    2. **Choose File Type:**
        - Select the type of document you want to upload (PDF, CSV, or Text).
    3. **Upload Documents:**
        - Use the file uploader to upload your documents.
    4. **Process and Generate Response:**
        - Click the "Process" button to analyze the documents.
    5. **Ask BookPal-AI:**
        - Type your questions in the text input box and receive responses.

    **Note:**
    Ensure that you've selected both an LLM model and uploaded documents before asking questions.

    ### Installation & Requirements
    1. **Clone the repository to your machine**
    2. Install the required packages and libraries by running the following command:
        pip install -r requirements.txt
    3. **Obtain API keys***
        go to https://platform.openai.com/api-keys and get your unique openAI api key, then go to https://huggingface.co/settings/tokens and get your unique instructor-xt key,
        lastly add them to the .env file in your project directory.
    4. **How to run**
        run the following command: streamlit run app.py
    
    ### Guides and resources used
        - https://youtu.be/dXxQ0LR-3Hg?si=6xF5dAfpCp0-famf
        - https://youtu.be/MlK6SIjcjE8?si=QZsBV4B2g4G3uJ_Q
        - https://youtu.be/RIWbalZ7sTo?si=_3SOScGec5Oy4GP0
"""


# PACKAGES
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# FUNCTIONS FOR DOCUMENT HANDINLING

def get_pdf_text(pdf_docs):
    # Extracts text from a list of PDF documents
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_docs):
    # Reads and decodes text from a list of CSV documents
    text = ""
    for csv in csv_docs:
        text += csv.read().decode('utf-8', errors='ignore')
    return text

def get_txt_text(txt_docs):
    # Reads and decodes text from a list of TXT documents
    text = ""
    for txt in txt_docs:
        text += txt.read().decode('utf-8', errors='ignore')
    return text

def get_txt_chunks(text):
    # Splits the input text into chunks using a specified text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, choice):
    # Creates a vector store from text chunks using specified embeddings
    if choice == "ChatOpenAI":
        embeddings = OpenAIEmbeddings()
    elif choice == "instructor-xl":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, choice):
    # Creates a conversation chain using a language model and a vector store
    if choice == "ChatOpenAI":
        llm = ChatOpenAI()
    elif choice == "instructor-xl":
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    # Handles user input, retrieves a response, and updates the chat history
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # Displaying the  response
    #USER = "user"
    #ASSISTANT = "assistant"
    #st.chat_message(USER).write(user_question)
    #st.chat_message(ASSISTANT).write(response['answer'])
    return response['answer']

# MAIN + STREAMLIT PAGE LAYOUT
    
def main():
    load_dotenv()
    st.set_page_config(page_title="BookPal-AI", page_icon="::books:")
    # Initialize conversation history if not present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Initialize memory for storing past questions and answers
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = []
    
    # Display chat messages
    user_question = st.chat_input("Ask BookPal a question...")
    st.title("BookPal-AI: Your LLM-powered Reading Companion")
    st.sidebar.header("Settings & Parameters")
    llm_choice = st.sidebar.selectbox("Choose an LLM Model", ["Models", "ChatOpenAI", "instructor-xl"])
    with st.sidebar:
        # file type selector
        file_type = st.radio("Choose file type", ["PDF", "CSV", "Text (txt)"])         

        # handles files of types .pdf
        if file_type == "PDF":
            # file uploader
            pdf_docs = st.file_uploader("Upload your PDF files here", accept_multiple_files=True, type=["pdf"])
            if not pdf_docs:  # handle if no pdf inserted
                pass
            elif pdf_docs:
                if st.button("Process"):
                    if llm_choice: # handle if no LLM chosen
                        with st.spinner("Processing PDFs..."):
                            # get txt
                            raw_text = get_pdf_text(pdf_docs)
                            # get txt chunks
                            text_chunks = get_txt_chunks(raw_text)
                            # create vector store
                            vector_store = get_vector_store(text_chunks, llm_choice)
                            # create conversation chain
                            st.session_state.conversation = get_conversation_chain(vector_store, llm_choice)
                            st.success("Done")
                    else:
                        st.warning("Please select an LLM model!")

        # handles files of types .txt
        elif file_type == "Text (txt)":
            # file uploader
            txt_docs = st.file_uploader("Upload your text files here", accept_multiple_files=True, type=["txt"])
            if not txt_docs:  # handle if no txt file inserted
                pass
            elif txt_docs :
                if st.button("Process"):
                    if llm_choice: # handle if no LLM chosen
                        with st.spinner("Processing text files..."):
                            # get txt
                            raw_text = get_txt_text(txt_docs)
                            # get txt chunks
                            text_chunks = get_txt_chunks(raw_text)
                            # create vector store
                            vector_store = get_vector_store(text_chunks, llm_choice)
                            # create conversation chain
                            st.session_state.conversation = get_conversation_chain(vector_store, llm_choice)
                            st.success("Done")
                    else:
                        st.warning("Please select an LLM model!")

        # handles files of types .csv
        elif file_type == "CSV":
            # file uploader
            csv_docs = st.file_uploader("Upload your CSV files here", accept_multiple_files=True, type=["csv"])
            if not csv_docs:  # handle if no csv file inserted
                pass
            elif csv_docs:
                if st.button("Process"):
                    if llm_choice: # handle if no LLM chosen
                        with st.spinner("Processing CSV files..."):
                            # get txt
                            raw_text = get_csv_text(csv_docs)
                            # get txt chunks
                            text_chunks = get_txt_chunks(raw_text)
                            # create vector store
                            vector_store = get_vector_store(text_chunks, llm_choice)
                            # create conversation chain
                            st.session_state.conversation = get_conversation_chain(vector_store, llm_choice)
                            st.success("Done")
                    else:
                        st.warning("Please select an LLM model!")

        #endif
            
    # endwith

    if user_question:
        if not file_type:  # handle if no document uploaded
            st.warning("Please upload a document.")
        else:
            if llm_choice:
                with st.spinner("Generating response..."):
                    # Save the current question and answer to memory
                    current_interaction = {'question': user_question, 'answer': handle_userinput(user_question)}
                    st.session_state.chat_memory.append(current_interaction)
            else:
                st.warning("Please select an LLM model!")

    # Display past questions and answers
    USER = "user"
    ASSISTANT = "assistant"
    for interaction in st.session_state.chat_memory:
        st.chat_message(USER).write(interaction['question'])
        st.chat_message(ASSISTANT).write(interaction['answer'])
    # endif
                
if __name__ == '__main__':
    main()
# endmain()