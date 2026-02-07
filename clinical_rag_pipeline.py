import os
import pandas as pd
from typing import List, Union
from dotenv import load_dotenv  # <-- Add this

# 1. Load the .env file
load_dotenv() 

# 2. Imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Ensure the key is loaded (it should be in os.environ now)
if not os.getenv("NVIDIA_API_KEY"):
    print("ERROR: NVIDIA_API_KEY not found in .env file!")

class ClinicalSystem:
    def __init__(self, csv_path, pdf_path):
        print("Initializing models...")
        # Llama 3.1 70B supports native tool calling
        self.llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
        self.embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
        
        print("Loading CSV...")
        self.df = pd.read_csv(csv_path)
        
        print("Indexing PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("System Ready.")

    def run(self, query: str):
        # Tool 1: PDF Search (Vector RAG)
        @tool
        def search_pdf(search_query: str):
            """Search clinical guidelines PDF for medical standards and protocols."""
            docs = self.vectorstore.similarity_search(search_query, k=3)
            return "\n".join([d.page_content for d in docs])

        # Tool 2: CSV Search (Structured Data)
        @tool
        def search_csv(search_query: str):
            """Query patient records CSV for specific data like A1C, demographics, or history."""
            agent = create_pandas_dataframe_agent(self.llm, self.df, allow_dangerous_code=True)
            return agent.run(search_query)

        tools = [search_pdf, search_csv]
        llm_with_tools = self.llm.bind_tools(tools)

        # Logic: LLM decides which tool to use
        messages = [
            SystemMessage(content="You are a clinical assistant. Use search_csv for patient data and search_pdf for medical guidelines. Synthesize a final answer based on the findings."), 
            HumanMessage(content=query)
        ]
        
        print("Agent is thinking and selecting tools...")
        ai_msg = llm_with_tools.invoke(messages)
        
        # Tool Execution Loop
        if ai_msg.tool_calls:
            messages.append(ai_msg)
            for tool_call in ai_msg.tool_calls:
                selected_tool = {"search_pdf": search_pdf, "search_csv": search_csv}[tool_call["name"]]
                print(f"Executing Tool: {tool_call['name']}...")
                tool_output = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
            
            # Final synthesized answer
            print("Synthesizing final answer...")
            final_resp = llm_with_tools.invoke(messages)
            return final_resp.content
        
        return ai_msg.content

if __name__ == "__main__":
    # Corrected Raw String Paths
    CSV = r"C:\Users\Vaishnavi Srivastava\Desktop\clinicalRag\mtsamples.csv"
    PDF = r"C:\Users\Vaishnavi Srivastava\Desktop\clinicalRag\medical.pdf"
    
    system = ClinicalSystem(CSV, PDF)
    
    # Hybrid Query Example
    q = "What are the first-line treatments for Type 2 Diabetes?"
    
    print("\n--- STARTING QUERY ---")
    result = system.run(q)
    print("\nFINAL CLINICAL RESPONSE:\n", result)