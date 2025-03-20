# **Intent-Based Query Routing System**

## **Overview**
This project is a **Python-based query routing system** that intelligently determines whether a user query should be processed using:
- **MySQL Database** (for structured queries)
- **Vector Database (Pinecone DB)** (for semantic searches in documents)

It leverages **LangChain, LangGraph**, and **Gemini API (1.5 Flash/2)** for intent classification and query execution.

## **Features**
✅ **Intent Classification:** Determines whether a query is related to SQL or requires a vector-based search.  
✅ **SQL Query Execution:** Runs queries on a **MySQL database** and retrieves structured data.  
✅ **Vector Database Search:** Performs similarity searches in **Pinecone DB** for unstructured data.  
✅ **Locally Installed LLM (Llama 2):** Used for processing SQL queries without relying on OpenAI.  
✅ **Optimized Query Handling:** Executes SQL queries with proper error handling and logging.  
✅ **API-Based Intent Classification:** Uses **Gemini 1.5 Flash/2 API** for determining query intent.  
✅ **Plain Text Responses:** Ensures that only the final result is displayed in the terminal.  

## **Project Structure**

📂 project-folder/ │── 📜 main.py # Manages intent classification and query routing │── 📜 sql.py # SQL agent for query generation and execution │── 📜 vectorDB_file_fetching.py # Handles vector database queries (Pinecone DB) │── 📜 requirements.txt # List of required dependencies │── 📜 README.md # Project documentation │── 📜 .env # Stores API keys (excluded from version control) │ └── 📂 data/ # (Optional) Directory for storing input files if needed


## **How It Works**
1. **User enters a query** in the terminal.
2. **Intent classifier** (via Gemini API) determines if it's an SQL or vector search query.
3. If **SQL query** → `sql.py` generates and executes a MySQL query.
4. If **vector search** → `vectorDB_file_fetching.py` retrieves relevant documents.
5. The **final answer is returned in plain text**, while logging/debugging messages are kept in `sql.py`.

## **Setup Instructions**
### **1. Install Dependencies**
Make sure you have **Python 3.9+** installed.

2. Set Up Environment Variables
Create a .env file in the project directory and add your API keys:
GEMINI_API_KEY=your_gemini_api_key_here

3. Run the Program
To start the query system, run:
python main.py

Logging and Debugging
* All logging and debugging messages are kept in sql.py for troubleshooting.
* The final output displayed in the terminal is only the plain text answer.
License
This project is open-source and available for modification. Feel free to contribute!


This README file provides a structured overview without exposing any credentials. Let me know if you need modifications! 🚀
