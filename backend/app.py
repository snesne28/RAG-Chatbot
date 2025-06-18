from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware



# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

app = FastAPI()

# Path to the folder containing PDFs
pdf_folder_path = "./documents"

# Functions for processing PDFs
def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are a helpful reading assistant who answers questions based on the snippets of text provided in context. Answer only using the context provided, being as concise as possible.
    If the information is not available, respond with: 
    "The answer is not available in the context provided."

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_tokens=10000)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Preload vector store
text = get_pdf_text_from_folder(pdf_folder_path)
text_chunks = get_text_chunks(text)
vector_store = get_vector_store(text_chunks)

# Add CORS middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI chatbot backend. Use the /ask endpoint to interact."}


@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            return JSONResponse(content={"error": "Question not provided"}, status_code=400)

        chain = get_conversational_chain()
        docs = vector_store.similarity_search(question)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answer_text = response.get('output_text', "No answer generated.")
        return {"answer": answer_text}
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

