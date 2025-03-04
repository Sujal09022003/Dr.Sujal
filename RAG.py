import os

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings

import streamlit as st

import chromadb
from chromadb.config import Settings

import json
import warnings
import time
from tqdm.auto import tqdm

DEVICE = "cpu"
warnings.filterwarnings('ignore')

class ChatGroqManager:
    def __init__(self):
        self.groq_api_key = "ur_Groq_API KEY"
        self.model_name = "llama-3.2-90b-vision-preview" 

        if not self.groq_api_key:
            raise ValueError("'GROQ_API_KEY not found in environment variables' or 'Not specified in backend'")
         

    def create_llm(self, temperature = 0.8):
        return ChatGroq(
            temperature=temperature,
            groq_api_key=self.groq_api_key,
            model_name=self.model_name
        )

def test_llm():
    try:
        groq_manager = ChatGroqManager()
        llm = groq_manager.create_llm()
        test_response = llm.invoke("Test connection - Who is the first Deputy-Prime Minister of India ??.")
        print(f"{test_response},LLM Connection Successful!")
        return True
    except Exception as e:
        st.error(f"LLM Connection Error: {str(e)}")
        return False

class SujalS_AgentKnowledge_on_Medicines:
    def __init__(self):
        with st.status("Initializing Assistant...", expanded=True) as status:
            try:
                # LLM Initialization
                print("Starting LLM initialization...")
                status.write("🔄 Initializing LLM connection...")
                self._initialize_llm()
                status.write("✅ LLM initialized successfully!")
                print("LLM initialization complete")
                
                # Embeddings Initialization
                print("Starting embeddings initialization...")
                status.write("🔄 Setting up embeddings model...")
                self._initialize_embeddings()
                status.write("✅ Embeddings model loaded!")
                print("Embeddings initialization complete")
                
                # Vectorstore Initialization
                print("Starting vectorstore initialization...")
                status.write("🔄 Loading knowledge base...")
                status.write("📚 Reading JSON files...")
                self._initialize_vectorstore()
                status.write("✅ Knowledge base loaded and indexed!")
                print("Vectorstore initialization complete")
                
                # Tools Setup
                print("Starting tools initialization...")
                status.write("🔄 Setting up QA tools...")
                self._initialize_tools()
                status.write("✅ Tools configured!")
                print("Tools initialization complete")
                
                # Agent Setup
                print("Starting agent initialization...")
                status.write("🔄 Finalizing agent setup...")
                self._initialize_agent()
                status.write("✅ Agent ready!")
                print("Agent initialization complete")
                
                status.update(label="✨ Assistant Ready!", state="complete")
            except Exception as e:
                print(f"Error during initialization: {str(e)}")
                status.update(label=f"Error: {str(e)}", state="error")
                raise

    def _initialize_llm(self):
        self.groq_manager = ChatGroqManager()
        self.llm = self.groq_manager.create_llm()

    def _initialize_embeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key = "OpenAPI Key")
    
    def _initialize_vectorstore(self):
        print("Checking for existing ChromaDB database...")
             
        chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="chroma_db"
        ))

        try:
            existing_collection = chroma_client.get_collection(name="pharma_knowledge_base")
            if existing_collection and existing_collection.count():
                print("Found existing ChromaDB collection with embeddings")
                self.vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="pharma_knowledge_base",
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing database with {existing_collection.count()} entries")
                return
        except ValueError:
            print("No existing collection found, creating new embeddings...")

        print("Starting to read JSON files...")
        docs = []
        data_dir = "microlabs_usa"
        file_count = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
        print(f"Found {file_count} JSON files to process")
        
        # Use tqdm for progress bar
        for filename in tqdm(os.listdir(data_dir), desc="Pulling aka Loading files"):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
                    docs.append(text)

        print("Starting text splitting...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        splits = text_splitter.create_documents(docs)
        total_chunks = len(splits)
        print(f"Created {total_chunks} text chunks")
        print("Creating new ChromaDB database...")
        
        batch_size = 64
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"Processing embeddings in {total_batches} batches on {DEVICE}...")
        
        # Show progress bar for embedding creation
        with tqdm(total=total_chunks, desc="Creating embeddings") as pbar:
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                
                # Create or update vectorstore
                if i == 0:
                    self.vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        client=chroma_client,
                        collection_name="pharma_knowledge_base"
                    )
                else:
                    self.vectorstore.add_documents(documents=batch)
                
                pbar.update(len(batch))
        print("ChromaDB database created successfully")
  
    def _initialize_tools(self):
        # Wrapper function to handle the RetrievalQA output format
        def qa_wrapper(query: str) -> str:
            result = self.qa(query)
            # Extract just the result text, ignoring source documents
            return result['result'] if isinstance(result, dict) else result

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        self.tools = [
            Tool(
                name="Product_Knowledge_Base",
                func=qa_wrapper,  # Use the wrapper function instead of self.qa.run
                description="Answer the Questions about pharmaceutical products and diseases."
            ),
            Tool(
                name="Summarizer",
                func=self.summarize,
                description="Generate 100 words summaries about Pharmaceuticals ."
            ),
            Tool(
                name="Recommender",
                func=self.recommend,
                description="Use this tool to provide recommendations based on symptoms or conditions."
            )
        ]

    def _initialize_agent(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
        )

    def summarize(self, product_name: str) -> str:
        context = self.vectorstore.similarity_search(product_name, k=4)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a concise summary of the pharmaceutical product."),
            ("human", "{product_name}\n\nContext: {context}")
        ])
        chain = prompt | self.llm
        return chain.invoke({
            "product_name": product_name,
            "context": "\n".join([doc.page_content for doc in context])
        })

    def recommend(self, query: str) -> str:
        context = self.vectorstore.similarity_search(query, k=4)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide recommendations based on the pharmaceutical knowledge base."),
            ("human", "{query}\n\nContext: {context}")
        ])
        chain = prompt | self.llm
        return chain.invoke({
            "query": query,
            "context": "\n".join([doc.page_content for doc in context])
        })

def main():
    st.set_page_config(
        page_title="SujalS' Pharmacy",
        page_icon="💊",
        layout="wide"
    )
    print("Starting main application...")

    st.title("")
    st.markdown("<h2 style='text-align: center; color: red;'>Dr.Sujal 😊 MBBS</h2>", unsafe_allow_html=True)
   

    with st.spinner("🔄 Testing LLM connection..."):
        print("Testing LLM connection...")
        if not test_llm():
            st.error("❌ Failed to connect to LLM. Please check your Groq-API Key, Use Groq-Cloud to make one.")
            st.stop()
            return
        print("LLM connection test successful")

    # Initialize the assistant
    if 'assistant' not in st.session_state:
        try:
            print("Starting worker initialization...")
            st.session_state.assistant = SujalS_AgentKnowledge_on_Medicines()
            st.session_state.messages = []
            print("Worker initialization completed successfully")
        except Exception as e:
            print(f"Failed to initialize worker: {str(e)}")
            st.error(f"❌ Failed to initialize worker: {str(e)}")
            st.stop()
            return

    #style and container on front-end     (Chatgpted)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            icon = "🧑‍💼" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"]):
                st.markdown(f"{icon} {message['content']}")

    if prompt := st.chat_input("Hi Namaste 🙏,I am worker in SujalS Pharmacy, ask Questions regarding Pharmaceuticals and Diseases:",max_chars=300):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"🧑‍💼 {prompt}")

        with st.chat_message("assistant"):
            try:
                with st.spinner('Processing your query Gentleman!!'):
                    response = st.session_state.assistant.agent.run(prompt)
                st.markdown(f"🤖 {response}")
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()