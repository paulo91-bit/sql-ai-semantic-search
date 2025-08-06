# -----------------------------------------------------------------
# FINAL SCRIPT: AI-POWERED SEMANTIC SEARCH WITH STANDARD OPENAI API
# -----------------------------------------------------------------
import os
import psycopg2
import openai
import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# --- 1. SETUP ---
# Load environment variables and configure the standard OpenAI client
def setup_ai_client():
    load_dotenv()
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ OpenAI client configured successfully.")
        return client
    except Exception as e:
        print(f"❌ Error configuring OpenAI client: {e}")
        return None

# --- 2. DATABASE FUNCTIONS ---

    
# Add this new import at the top of your SqlAI.py file


# Function to get a new database connection (UPDATED)
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        # Register the vector type adapter
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to the database. Is the Docker container running? Error: {e}")
        return None

# Function to set up the table
def setup_database_table():
    conn = get_db_connection()
    if not conn: return
    
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS product_reviews (
                id SERIAL PRIMARY KEY,
                review_text TEXT,
                embedding VECTOR(1536)
            );
            """)
        conn.commit()
        print("✅ Database table is ready.")
    except Exception as e:
        print(f"❌ Error setting up database table: {e}")
    finally:
        if conn: conn.close()

# Function to embed and store reviews
def embed_and_store_reviews(client):
    reviews = [
        "This is the best moisturizer I have ever used, my face feels so smooth.",
        "The battery life on this phone is amazing, it lasts for two full days.",
        "A fantastic book, the plot was gripping from start to finish.",
        "I love this coffee maker, it's fast and the coffee tastes great.",
        "My skin felt irritated and red after using this serum."
    ]
    
    conn = get_db_connection()
    if not conn: return
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM product_reviews;")
            if cur.fetchone()[0] == 0:
                print("Embedding new reviews...")
                for review in reviews:
                    response = client.embeddings.create(input=review, model="text-embedding-3-small")
                    embedding = response.data[0].embedding
                    cur.execute("INSERT INTO product_reviews (review_text, embedding) VALUES (%s, %s)", (review, np.array(embedding)))
                conn.commit()
                print(f"✅ {len(reviews)} reviews have been embedded and stored.")
            else:
                print("ℹ️ Reviews already exist in the database, skipping insertion.")
    except Exception as e:
        print(f"❌ Error during embedding or insertion: {e}")
    finally:
        if conn: conn.close()

# Function for semantic search
def find_similar_reviews(client, query, limit=3):
    conn = get_db_connection()
    if not conn: return

    try:
        print(f"\n▶️  Searching for reviews similar to: '{query}'")
        response = client.embeddings.create(input=query, model="text-embedding-3-small")
        query_embedding = response.data[0].embedding
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT review_text, 1 - (embedding <=> %s) AS similarity 
                FROM product_reviews 
                ORDER BY similarity DESC
                LIMIT %s;
            """, (np.array(query_embedding), limit))
            results = cur.fetchall()

        if not results:
            print("No similar reviews found.")
            return

        print("Top results:")
        for row in results:
            print(f"  - '{row[0]}' (Similarity: {row[1]:.2f})")
    except Exception as e:
        print(f"❌ Error during search: {e}")
    finally:
        if conn: conn.close()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    ai_client = setup_ai_client()
    
    if ai_client:
        setup_database_table()
        embed_and_store_reviews(ai_client)
        
        find_similar_reviews(ai_client, "a product for my face")
        find_similar_reviews(ai_client, "a good story")
        find_similar_reviews(ai_client, "a device with long power")
        
        print("\n✅ Script finished.")