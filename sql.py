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
    table_name: str
    query: str
    result: str
    answer: str
    error: str

# Initialize Llama 3 Model from Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# ✅ Define SQL Query Output Schema
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# ✅ Define Table Schemas
##to decrease the length of the code you can take the schemas part in schemas.py, a separate code file.
TABLE_SCHEMAS = {
    "music_dataset": """
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
        | `Tempo (BPM)`       | int  |
        | Mood                | text |
        | Energy              | text |
        | Danceability        | text |
        +---------------------+------+
    """,
    "employee": """
        Table: employee
        +---------------+------+
        | Column        | Type |
        +---------------+------+
        | `Index`       | int  |
        | `First Name`  | text |
        | `Last Name`   | text |
        | Sex           | text |
        | Email         | text |
        | Phone         | text |
        | `Date of birth` | text |
        | `Job Title`   | text |
        +---------------+------+
    """
}

# ✅ Improved Table Selection - Add keyword-based approach as a fallback
def determine_table(state: State):
    """Determines which table to query based on the user question."""
    question = state["question"].lower()
    print(f"🔍 Determining table for question: {question}")
    
    # Quick keyword-based matching for efficiency
    employee_keywords = ["employee", "name", "job", "email", "phone", "gender", "sex", "birth"]
    music_keywords = ["music", "song", "artist", "genre", "tempo", "mood", "energy", "dance"]
    
    # Check if any keywords match
    employee_matches = sum(1 for keyword in employee_keywords if keyword in question)
    music_matches = sum(1 for keyword in music_keywords if keyword in question)
    
    if employee_matches > music_matches:
        table_name = "employee"
        print(f"📊 Selected table via keyword match: {table_name} (keyword match)")
        return {"table_name": table_name}
    elif music_matches > employee_matches:
        table_name = "music_dataset"
        print(f"📊 Selected table via keyword match: {table_name} (keyword match)")
        return {"table_name": table_name}
    
    # Use LLM to determine the table if keyword matching is inconclusive
    prompt = f"""
You are tasked with determining which database table is most appropriate for answering a user's question.

User question: {question}

Available tables and their schemas:
{TABLE_SCHEMAS["music_dataset"]}

{TABLE_SCHEMAS["employee"]}

Based ONLY on the user's question, which table should be queried?
Answer with ONLY ONE word - either "music_dataset" or "employee".
"""
    try:
        response = llm.invoke(prompt)
        llm_response = response.content.strip().lower()
        
        # Validate the response
        if "employee" in llm_response:
            table_name = "employee"
        elif "music" in llm_response or "dataset" in llm_response or "music_dataset" in llm_response:
            table_name = "music_dataset"
        else:
            # Default to most relevant based on keywords if LLM is ambiguous
            table_name = "employee" if employee_matches >= music_matches else "music_dataset"
            print(f"⚠️ Unclear LLM response: '{llm_response}'. Defaulting to {table_name}.")
    except Exception as e:
        print(f"⚠️ Error getting table from LLM: {e}. Using keyword matching.")
        table_name = "employee" if employee_matches >= music_matches else "music_dataset"
    
    print(f"📊 Selected table: {table_name}")
    return {"table_name": table_name}

# ✅ Extract SQL Query from LLM response
def extract_sql_query(response_text):
    """Extract the actual SQL query from LLM response text."""
    # Check if the response is wrapped in markdown code blocks
    code_pattern = re.compile(r'```(?:sql)?(.*?)```', re.DOTALL)
    code_match = code_pattern.search(response_text)
    
    if code_match:
        # Extract just the SQL inside the code blocks
        query = code_match.group(1).strip()
    else:
        # If not in code blocks, try to extract SELECT statement
        select_pattern = re.compile(r'(SELECT[\s\S]*?;)', re.IGNORECASE)
        select_match = select_pattern.search(response_text)
        if select_match:
            query = select_match.group(1).strip()
        else:
            # If no clear SQL syntax is found, use the entire response
            query = response_text.strip()
    
    # Remove any trailing "LIMIT" if it appears after the semicolon
    query = re.sub(r';[\s\S]*', ';', query)
    
    # Add LIMIT if it doesn't exist
    if "LIMIT" not in query.upper():
        # If query ends with semicolon, insert before it
        if query.rstrip().endswith(';'):
            query = query.rstrip()[:-1] + " LIMIT 10;"
        else:
            query = query.rstrip() + " LIMIT 10;"
    
    return query

# ✅ Generate SQL Query
def write_query(state: State, max_retries=3):
    """Generate SQL query to fetch information based on user question."""
    print(f"📝 Generating SQL query for table: {state['table_name']}")
    
    user_question = state["question"]
    table_name = state["table_name"]
    attempt = 0
    sql_query = ""
    
    while attempt < max_retries:
        try:
            prompt_text = f"""
You are an expert SQL assistant. Generate a MySQL query based on the user's question.

### User Question: {user_question}

### Database Schema:
{TABLE_SCHEMAS[table_name]}

### **CRITICAL INSTRUCTIONS:**
1. ONLY use columns that exist in the schema above.
2. For column names with spaces, ALWAYS use backticks around the ENTIRE column name.
   - CORRECT: `First Name`
   - INCORRECT: First Name
   - INCORRECT: `First` Name
3. Use `COLLATE utf8mb4_general_ci` for case-insensitive string comparisons.
4. Use `LIKE '%term%'` for partial matching.
5. Use `LIMIT 10` unless the user specifies a different limit.
6. The query MUST be for the {table_name} table only.
7. Return ONLY the SQL query itself with no introduction or explanation.
8. Do not include markdown code blocks (```).
9. Do not include phrases like "Here is the query:" or any other text.
"""
            result = llm.invoke(prompt_text)
            response_text = result.content.strip()
            
            # Extract just the SQL query from the response
            extracted_query = extract_sql_query(response_text)
            
            # Verify the response contains a SQL query
            if "SELECT" in extracted_query.upper():
                # Additional fix to ensure backticks are properly placed
                sql_query = fix_column_backticks(extracted_query, table_name)
                print(f"✅ Generated query: {sql_query}")
                break
            else:
                print(f"⚠️ Invalid query (Attempt {attempt+1}): {extracted_query}")
                attempt += 1
        except Exception as e:
            print(f"❌ Error during query generation (Attempt {attempt+1}): {e}")
            attempt += 1

    if not sql_query:
        error_msg = "Failed to generate a valid SQL query after multiple attempts."
        print(f"❌ {error_msg}")
        return {"query": "", "error": error_msg}

    return {"query": sql_query, "error": ""}

# ✅ Helper function to fix column name backticks
def fix_column_backticks(query, table_name):
    """Fix backticks around column names with spaces."""
    # Get column names that need backticks (those with spaces)
    columns_with_spaces = []
    for line in TABLE_SCHEMAS[table_name].split('\n'):
        line = line.strip()
        if '|' in line and 'Column' not in line and '+--' not in line:
            # Extract column name
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 1:
                col_name = parts[1].strip()
                if ' ' in col_name and not (col_name.startswith('`') and col_name.endswith('`')):
                    columns_with_spaces.append(col_name)
    
    # Fix query for each column that should have backticks
    fixed_query = query
    for col in columns_with_spaces:
        # Replace unquoted occurrences of the column name
        pattern = r'(?<![`\w])' + re.escape(col) + r'(?![`\w])'
        replacement = f'`{col}`'
        fixed_query = re.sub(pattern, replacement, fixed_query)
    
    # Additional fix for specific column patterns found in the error
    problematic_patterns = [
        (r'`First` Name', '`First Name`'),
        (r'`Last` Name', '`Last Name`'),
        (r'`Job` Title', '`Job Title`'),
        (r'`Date` of birth', '`Date of birth`')
    ]
    
    for pattern, replacement in problematic_patterns:
        fixed_query = fixed_query.replace(pattern, replacement)
    
    return fixed_query

# ✅ Execute SQL Query with Error Handling
def execute_query(state: State):
    """Execute the generated SQL query and return results."""
    # Check if there's already an error
    if state.get("error"):
        print(f"⚠️ Skipping query execution due to previous error: {state['error']}")
        return {"result": "", "error": state["error"]}
    
    print(f"🔄 Executing query: {state['query']}")
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        print(f"✅ Query executed successfully.")
        return {"result": result, "error": ""}
    except sqlalchemy.exc.SQLAlchemyError as e:
        error_msg = f"Query execution failed with SQLAlchemy error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Try to fix common syntax errors and retry
        if "syntax" in str(e).lower():
            try:
                fixed_query = fix_query_syntax(state["query"], state["table_name"])
                if fixed_query != state["query"]:
                    print(f"🔄 Retrying with fixed query: {fixed_query}")
                    result = execute_query_tool.invoke(fixed_query)
                    print(f"✅ Fixed query executed successfully.")
                    return {"query": fixed_query, "result": result, "error": ""}
            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
        
        return {"result": "", "error": error_msg}
    except Exception as e:
        error_msg = f"Query execution failed with unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"result": "", "error": error_msg}

# ✅ Helper function to fix common query syntax errors
def fix_query_syntax(query, table_name):
    """Fix common SQL syntax errors."""
    fixed_query = query
    
    # Add table name to FROM clause if missing
    if "FROM" in fixed_query.upper() and table_name not in fixed_query:
        fixed_query = fixed_query.replace("FROM", f"FROM {table_name}")
    
    # Fix missing backticks around column names with spaces
    for column in ["First Name", "Last Name", "Date of birth", "Job Title", "Tempo (BPM)"]:
        if column in fixed_query and f"`{column}`" not in fixed_query:
            fixed_query = fixed_query.replace(column, f"`{column}`")
    
    # Fix incorrect backtick usage
    for wrong, right in [
        ("`First` Name", "`First Name`"),
        ("`Last` Name", "`Last Name`"),
        ("`Job` Title", "`Job Title`"),
        ("`Date` of birth", "`Date of birth`")
    ]:
        fixed_query = fixed_query.replace(wrong, right)
    
    # Fix COLLATE placement (it should be after the WHERE condition)
    collate_wrong = "WHERE COLLATE utf8mb4_general_ci"
    if collate_wrong in fixed_query:
        fixed_query = fixed_query.replace(collate_wrong, "WHERE")
    
    # Move misplaced COLLATE to the correct position (inside LIKE conditions)
    if "COLLATE utf8mb4_general_ci" in fixed_query and "LIKE" in fixed_query:
        # Remove standalone COLLATE
        fixed_query = re.sub(r'\sCOLLATE utf8mb4_general_ci(?![)])', ' ', fixed_query)
        
        # Add COLLATE to each LIKE condition that doesn't already have it
        like_pattern = r'(LIKE \'\%.*?\%\')'
        like_matches = re.finditer(like_pattern, fixed_query, re.IGNORECASE)
        for match in like_matches:
            like_expr = match.group(1)
            if "COLLATE" not in like_expr:
                replacement = f"{like_expr} COLLATE utf8mb4_general_ci"
                fixed_query = fixed_query.replace(like_expr, replacement)
    
    # Make sure the query ends with a semicolon
    if not fixed_query.rstrip().endswith(';'):
        fixed_query = fixed_query.rstrip() + ';'
    
    return fixed_query

# ✅ Generate Answer from SQL Query Result
def generate_answer(state: State):
    """Generate a natural language response from SQL query results."""
    # Check if there's an error
    if state.get("error"):
        answer = f"I encountered an error while trying to answer your question: {state['error']}. Please try rephrasing your question."
        return {"answer": answer}
    
    print("🧠 Generating natural language answer from query results")
    prompt = f"""
As an AI assistant, provide a helpful and natural response to the user's question based on the SQL query and its results.

User Question: {state["question"]}
Selected Table: {state["table_name"]}
SQL Query: {state["query"]}
SQL Result: {state["result"]}

Guidelines:
1. If the result is empty, explain that no matching records were found.
2. Format the response in a clear, easy-to-read manner.
3. If appropriate, include a brief summary of the data.
4. Be conversational and helpful in your tone.
5. DO NOT mention technical details like SQL queries unless absolutely necessary.
6. Keep your response concise and direct.
"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        print(f"✅ Generated answer.")
        return {"answer": answer}
    except Exception as e:
        error_msg = f"Failed to generate answer: {str(e)}"
        print(f"❌ {error_msg}")
        return {"answer": f"I found some information but encountered an error while formatting the response. Error: {str(e)}"}

# ✅ Build Improved Execution Graph
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("determine_table", determine_table)
    graph_builder.add_node("write_query", write_query)
    graph_builder.add_node("execute_query", execute_query)
    graph_builder.add_node("generate_answer", generate_answer)
    
    # Define the flow
    graph_builder.add_edge(START, "determine_table")
    graph_builder.add_edge("determine_table", "write_query")
    graph_builder.add_edge("write_query", "execute_query")
    graph_builder.add_edge("execute_query", "generate_answer")
    
    return graph_builder.compile()

# Create the graph
graph = build_graph()

def handle_query(user_question: str):
    """Handles query execution through LangGraph and returns plain text answer."""
    initial_state = {
        "question": user_question,
        "table_name": "",
        "query": "",
        "result": "",
        "answer": "",
        "error": ""
    }
    
    print(f"🚀 Processing query: {user_question}")
    result = graph.invoke(initial_state)
    
    if result.get("error"):
        print(f"❌ Query processing completed with error: {result['error']}")
    else:
        print("✅ Query processing completed successfully")
    
    return result.get("answer", "Sorry, I was unable to generate an answer. Please try again.")

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        answer = handle_query(user_input)
        print("\nResult:")
        print(answer)
        print()
