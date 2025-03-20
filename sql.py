import os
import getpass
import re
import sqlalchemy
from typing_extensions import TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain.prompts import SystemMessagePromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from IPython.display import display, Image

# ✅ Secure API Key Input
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# ✅ Connect to MySQL Database
try:
    db = SQLDatabase.from_uri("mysql+pymysql://username:password@localhost/database_name")
    print("✅ Connected to:", db.dialect)
    print("📂 Tables:", db.get_usable_table_names())
except Exception as e:
    print(f"❌ Database connection error: {e}")
    exit()

# ✅ Define State Structure
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# Initialize Llama 3 Model from Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# ✅ Custom SQL Query Prompt (Better Column Handling)
query_prompt_template = """
You are an expert SQL assistant. Generate a MySQL query based on the user's question.

Special Query Rules:
1. If the question includes "song", search in `Song_Name` column.
2. If the question includes "artist" or "singer", search in `Artist` column.
3. **Extract the artist name from the user's question and use it in the query.**
4. **Do NOT default to "Coldplay" unless explicitly mentioned.**
5. Use `COLLATE utf8mb4_general_ci` for case-insensitive matching.
6. Use `LIMIT {top_k}` unless a specific number is requested.

### Example Queries:
- **User:** "How many songs are there?"
  - **Query:** `SELECT COUNT(DISTINCT Song_Name) FROM music_dataset;`

- **User:** "Show top 5 songs of Coldplay"
  - **Query:** `SELECT DISTINCT Song_Name FROM music_dataset WHERE Artist COLLATE utf8mb4_general_ci = 'Coldplay' LIMIT 5;`

- **User:** "Name one song of Kanye West"
  - **Query:** `SELECT DISTINCT Song_Name FROM music_dataset WHERE Artist COLLATE utf8mb4_general_ci = 'Kanye West' LIMIT 1;`
"""




# ✅ Define SQL Query Output Schema
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

import re

def extract_artist_name(question):
    """Extracts the artist name from a user query before sending it to the LLM."""
    match = re.search(r"(?i)(?:songs? (?:of|by)|by) ([\w\s]+)", question)
    if match:
        return match.group(1).strip()
    return None




# ✅ Generate SQL Query
def write_query(state: State, max_retries=3):
    """Generate SQL query to fetch information based on user question with schema awareness."""
    print("📝 Received user question:", state["question"])  # Debugging
    
    user_question = state["question"]  # Extract the user's question
    attempt = 0
    sql_query = ""
    
    # Define the table schema information
    schema_info = """
Table: music_dataset
+---------------------+------+
| Column              | Type |
+---------------------+------+
| User_ID             | text |
| User_Text           | text |
| Sentiment_Label     | text |
| Recommended_Song_ID | text |
| Song_Name           | text |
| Artist              | text |
| Genre               | text |
| `Tempo (BPM)`       | int  |  -- Note: This column name requires backticks in SQL
| Mood                | text |
| Energy              | text |
| Danceability        | text |
+---------------------+------+
"""
    
    while attempt < max_retries:
        try:
            prompt_text = f"""
You are an expert SQL assistant. Generate a MySQL query based on the user's question.

### User Question: {user_question}

### Database Schema:
{schema_info}

### **Rules:**
1. ONLY use columns that exist in the schema above.
2. If the question is about sentiment, use the `Sentiment_Label` column.
3. If the question includes "song", search in `Song_Name` column.
4. If the question includes "artist" or "singer", search in `Artist` column.
5. For mood-related queries, use the `Mood` column.
6. For tempo queries, use the `Tempo (BPM)` column with proper backticks: `Tempo (BPM)`.
7. **IMPORTANT: Column names with spaces or special characters must be enclosed in backticks.**
8. **Extract any names (artist, song) from the user's question and use them in the query.**
9. **Use `COLLATE utf8mb4_general_ci` for case-insensitive comparisons.**
10. **Use `LIKE '%term%'` for partial matching.**
11. **Use `LIMIT 10` unless the user specifies a different limit.**

### **Example Queries**
- **User:** "How many songs are there?"  
  **Query:** SELECT COUNT(DISTINCT Song_Name) FROM music_dataset;

- **User:** "Show top 5 songs of Coldplay"  
  **Query:** SELECT DISTINCT Song_Name FROM music_dataset WHERE Artist LIKE '%Coldplay%' COLLATE utf8mb4_general_ci LIMIT 5;

- **User:** "Tell me sentiment of the song Uptown Funk"  
  **Query:** SELECT Song_Name, Artist, Sentiment_Label FROM music_dataset WHERE Song_Name LIKE '%Uptown Funk%' COLLATE utf8mb4_general_ci LIMIT 1;

- **User:** "What's the mood of songs by Taylor Swift?"  
  **Query:** SELECT Song_Name, Mood FROM music_dataset WHERE Artist LIKE '%Taylor Swift%' COLLATE utf8mb4_general_ci LIMIT 10;

- **User:** "What is the tempo of the song Happy?"
  **Query:** SELECT Song_Name, Artist, `Tempo (BPM)` FROM music_dataset WHERE Song_Name LIKE '%Happy%' COLLATE utf8mb4_general_ci LIMIT 1;

IMPORTANT: Return ONLY the SQL query with no explanations, comments, or backticks around the entire query.
Make sure to use backticks around column names with spaces or special characters like `Tempo (BPM)`.
"""

            print("🔎 Sending to LLM:", prompt_text)  # Debugging

            result = llm.invoke(prompt_text)
            
            # Clean up the response to extract just the SQL query
            response_text = result.content.strip()
            
            # Try to extract query if it's wrapped in backticks or code blocks
            if "```sql" in response_text:
                match = re.search(r"```sql\n(.*?)\n```", response_text, re.DOTALL)
                if match:
                    sql_query = match.group(1).strip()
            elif "```" in response_text:
                match = re.search(r"```\n?(.*?)\n?```", response_text, re.DOTALL)
                if match:
                    sql_query = match.group(1).strip()
            else:
                # If no code blocks, just use the entire response
                # But first, remove any explanatory text
                lines = response_text.splitlines()
                cleaned_lines = []
                for line in lines:
                    if any(keyword in line.upper() for keyword in ["SELECT", "FROM", "WHERE", "LIMIT", "ORDER", "GROUP", "HAVING", "JOIN"]):
                        cleaned_lines.append(line)
                
                if cleaned_lines:
                    sql_query = " ".join(cleaned_lines)
                else:
                    sql_query = response_text

            # Final check to ensure we have a valid query
            if sql_query and "SELECT" in sql_query.upper():
                print(f"✅ Valid query generated: {sql_query}")
                break
            else:
                print(f"⚠️ Invalid query (Attempt {attempt+1}): {sql_query}")
                attempt += 1
        except Exception as e:
            print(f"❌ Error (Attempt {attempt+1}): {e}")
            attempt += 1

    if not sql_query or "SELECT" not in sql_query.upper():
        raise ValueError("Failed to generate a valid SQL query after multiple attempts.")

    return {"query": sql_query}
# ✅ Execute SQL Query with Error Handling
def execute_query(state: State):
    """Execute the generated SQL query and return results."""
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        return {"result": result}
    except sqlalchemy.exc.SQLAlchemyError as e:
        print(f"🔴 SQLAlchemy Error: {e}")
        return {"result": f"❌ Query execution failed: {str(e)}"}
    except Exception as e:
        print(f"🔴 Unknown Error: {e}")
        return {"result": f"❌ Query execution failed: {str(e)}"}

# ✅ Generate Answer from SQL Query Result
def generate_answer(state: State):
    """Generate a natural language response from SQL query results."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )

    response = llm.invoke(prompt)
    return {"answer": response.content}

# ✅ Build Execution Graph
graph_builder = StateGraph(State)
graph_builder.add_edge(START, "write_query")
graph_builder.add_sequence([write_query, execute_query, generate_answer])
graph = graph_builder.compile()

# ✅ Display Execution Graph
display(Image(graph.get_graph().draw_mermaid_png()))

# ✅ Function to Ask Query in Terminal
def ask_query():
    """Ask a query from the user in the terminal and process it."""
    user_question = input("\nEnter your query: ").strip()  # Remove whitespace

    if not user_question:
        print("⚠️ Query cannot be empty. Please enter a valid question.")
        return  # Exit early

    print("\nProcessing query... 🔄")

    state = {"question": user_question}

    for step in graph.stream(state, stream_mode="updates"):
        print("Step output:", step)
        if "query" in step:
            print("📝 Generated SQL Query:", step["query"])
        elif "result" in step:
            print("📊 Query Results:", step["result"])
        elif "answer" in step:
            print("✅ Final Answer:", step["answer"])

# ✅ Run Query Function for `main.py`
def handle_query(user_question: str):
    """Handles query execution through LangGraph and returns plain text answer."""
    state = {"question": user_question}
    result = graph.invoke(state)

    # Extract and return only the plain answer
    return result.get("answer", "❌ No answer generated.")
