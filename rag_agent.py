from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

# 1. Load PDF
loader = PyPDFLoader("acmecorp-employee-handbook.pdf")
data = loader.load()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(data)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
vector_store = InMemoryVectorStore(embeddings)

# 5. Add documents
vector_store.add_documents(documents=all_splits)

# 6. Create search tool
@tool
def search_handbook(query: str) -> str:
    """Search information inside the employee handbook PDF."""
    results = vector_store.similarity_search(query)
    return results[0].page_content

# 7. Model
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 8. Agent
agent = create_agent(
    model=model,
    tools=[search_handbook],
    system_prompt="You are a helpful agent that answers using the PDF content."
)

# 9. Ask question
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="How many days of vacation does an employee get in their first year?")
        ]
    }
)

print(response["messages"][-1].content)