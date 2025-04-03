import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def classify_intent(query: str) -> str:
    """Classify user intent as either 'sql' or 'vector_db' using Gemini API."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

    prompt = f"""
    You are an intent classifier. Determine if the query is related to 'SQL' or 'VectorDB'.
    
    Rules:
    - If the query is about music-related topics (songs, artists, genres), return **ONLY** "SQL".
    - If the query is about general information or non-music topics (AI, laws, metabolism), return **ONLY** "VectorDB".

    Return exactly one of these words: **"SQL"** or **"VectorDB"** (without extra text).

    User query: "{query}"
    """

    response = llm.invoke(prompt).content.strip()
    logging.info(f"Raw Intent Response from Gemini: {response}")

    intent = response.lower().replace(" ", "").strip()  # Normalize response
    print("the Intent is ==>" + intent)  # Debugging print

    # Ensure correct classification
    if "sql" in intent:
        return "sql"
    elif "vectordb" in intent or "vector_db" in intent or "vector" in intent:
        return "vector_db"
    else:
        return "unknown"


    

def process_query(query: str):
    """Route the query based on intent classification."""
    intent = classify_intent(query)
    logging.info(f"Intent classified as: {intent}")

    if intent == "sql":
        try:
            import sql  # Import inside function to avoid unnecessary loading
            logging.info("Routing query to SQL Agent...")
            result = sql.handle_query(query)  # Ensure sql.py has handle_query()
        except Exception as e:
            logging.error(f"Error processing SQL query: {e}")
            result = "An error occurred while processing the SQL query."

    elif intent == "vector_db":
        try:
            import vectorDB_file_fetching  # Import inside function to avoid unnecessary loading
            logging.info("Routing query to Vector Database...")
            result = vectorDB_file_fetching.handle_query(query)  # Ensure vectorDB_file_fetching.py has handle_query()
        except Exception as e:
            logging.error(f"Error processing Vector DB query: {e}")
            result = "An error occurred while searching the Vector Database."

    else:
        result = "Could not determine intent. Please refine your query."

    print("\nResult:")
    print(result)

def main():
    """Main loop for user interaction."""
    print("Welcome! Type your query or type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break  
        
        process_query(user_input)

if __name__ == "__main__":
    main()
