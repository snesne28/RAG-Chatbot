Info-Bot (RAG + Google Gemini)
This project is an intelligent chatbot for my college website that uses Retrieval-Augmented Generation (RAG) and Google Gemini LLM to answer any queries related to the institution. It processes PDF documents containing web scraped data and provides accurate, context-based responses in natural language.

📌 Project Description

The chatbot enables users—students, faculty, or visitors—to ask questions about the college and receive responses grounded in actual college documents. This ensures reliable and factual information delivery.

It integrates:

  Document parsing from PDFs
  Embedding & storage of content using FAISS
  Query answering with context retrieval + generative AI

⚙️ How It Works

        1. 📄 Document Ingestion
        PDFs are read using PyPDF2.
        Text is split into meaningful chunks using LangChain’s RecursiveCharacterTextSplitter.
        
        2. 📊 Vectorization
        Each chunk is embedded using Google Gemini Embedding Model (embedding-001).
        The embeddings are stored in a FAISS vector database for efficient similarity search.
        
        3. ❓ Query Handling (RAG)
        On receiving a user question, the bot searches the FAISS vector store to retrieve relevant chunks.
        These are passed to the gemini-pro model for response generation.

        4. 💬 Response Generation
        Gemini LLM generates an answer grounded in the retrieved documents.
        If no context matches, the bot replies:
        "The answer is not available in the context provided."

🧠 Tech Stack
       
| Layer                      | Technology                                           |
| -------------------------- | ---------------------------------------------------- |
|   Frontend                 | Any (can be integrated with React, plain HTML, etc.) |
|   Backend API              |  FastAPI                                             |
|   LLM & Embedding          |  Google Gemini (gemini-pro , embedding-001)          |
|   PDF Parsing              |  PyPDF2                                              |
|   Vector Store             |  FAISS via langchain_community.vectorstores          |
|   RAG Framework            |  LangChain                                           |
|   Environment Management   |  python-dotenv                                       |

🚀 FastAPI Endpoints

      GET /
        Returns a welcome message or basic project info.
      POST /ask
        Accepts a JSON with a user query. Returns a grounded response.

🔐 Environment Variables

      The .env file must include:

      GOOGLE_API_KEY=your_google_generative_ai_key

🛠️ Dependencies (requirements.txt)

        fastapi
        uvicorn
        PyPDF2
        langchain
        langchain_community
        langchain_google_genai
        python-dotenv
        google-generativeai

📌Features

        ✅ Accurate, document-grounded answers
        ✅ Scalable to new document types and formats
        ✅ Easy integration with frontend interfaces
        ✅ CORS-enabled backend for flexible deployment

📚 Use Cases

        Answering student and parent queries on college websites
        Virtual helpdesk for admissions, hostel, academics, etc.
        Research assistant for browsing large policy or curriculum documents

✍️ Author

        Sneha Prakash.
        Built as part of an academic project using state-of-the-art LLM technologies.
