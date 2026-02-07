import streamlit as st
import os
from dotenv import load_dotenv
from clinical_rag_pipeline import ClinicalSystem # Assuming your logic is in this file

# 1. Page Configuration
st.set_page_config(page_title="Clinical AI Assistant", page_icon="🏥", layout="wide")
st.title("🏥 Clinical RAG Reasoning System")
st.markdown("---")

# 2. Load environment & check API Key
load_dotenv()
if not os.getenv("NVIDIA_API_KEY"):
    st.error("NVIDIA_API_KEY not found! Please check your .env file.")
    st.stop()

# 3. Initialize the Pipeline (Cached so it doesn't reload every time you click)
@st.cache_resource
def init_system():
    # Paths to your files
    CSV_PATH = r"C:\Users\Vaishnavi Srivastava\Desktop\clinicalRag\mtsamples.csv"
    PDF_PATH = r"C:\Users\Vaishnavi Srivastava\Desktop\clinicalRag\medical.pdf"
    return ClinicalSystem(CSV_PATH, PDF_PATH)

with st.spinner("Initializing Clinical Engine... (Loading PDF & CSV)"):
    try:
        clinical_bot = init_system()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.stop()

# 4. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input
if prompt := st.chat_input("Ask a clinical question (e.g., 'Compare patient 101 to the guidelines')"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("🔍 *Thinking and checking records...*")
        
        try:
            # Call the run method from your class
            response = clinical_bot.run(prompt)
            status_placeholder.empty()
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {e}")