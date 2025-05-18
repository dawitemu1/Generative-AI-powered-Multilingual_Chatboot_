import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langdetect import detect
import os

# --------------------- Configuration ---------------------
os.environ["TRANSFORMERS_NO_TF"] = "1"
FAISS_DB_PATH = "faiss_faqs_db_Amh"

# --------------------- Load LLM & Vector Store ---------------------
llm = Ollama(model="mannix/llama3-8b-ablitered-v3:iq4_xs")
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

if os.path.exists(FAISS_DB_PATH):
    try:
        vector_store = FAISS.load_local(
            FAISS_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever()
    except Exception as e:
        st.error(f"❌ Error loading FAISS vector store: {e}")
        st.stop()
else:
    st.error("❌ FAISS database not found. Please make sure the vector store is prepared.")
    st.stop()

# --------------------- QA Chain ---------------------
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --------------------- Streamlit Chatbot UI ---------------------
st.set_page_config(page_title="CBE Chatbot", page_icon="💬")
st.title("💬 CBE AI Chatbot")
st.caption("Ask your questions in English or Amharic about the Commercial Bank of Ethiopia.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# Chat input box
user_message = st.chat_input("Type your question in English or Amharic...")

# Response generation
if user_message:
    st.chat_message("user").markdown(user_message)

    try:
        # Detect input language
        lang = detect(user_message.strip())

        # Create system prompt based on detected language
        if lang == "am":
            system_prompt = (
                "አንተ ለኮሜርሻል ባንክ ኢትዮጵያ (CBE) ቻትቦት የተማረ አጋዥ ነህ። "
                "ጥያቄዎችን በግልጽ፣ በአጭር እና በተመጣጣኝ መልኩ በአማርኛ መልስ ስጥ።\n\n"
                f"ተጠቃሚ፡ {user_message.strip()}\nየአጋዥ መልስ፡"
            )
        else:
            system_prompt = (
                "You are a helpful assistant trained on Commercial Bank of Ethiopia's Chatbot powered by LLaMA3. "
                "Respond clearly and concisely in English.\n\n"
                f"User: {user_message.strip()}\nAssistant:"
            )

        # Run QA chain
        response = qa_chain.run(system_prompt)

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(f"💡 **Answer:** {response}")

        # Update chat history
        st.session_state.chat_history.append({
            "user": user_message.strip(),
            "bot": f"💡 **Answer:** {response}"
        })

    except Exception as e:
        st.error(f"⚠️ Error generating response: {e}")
