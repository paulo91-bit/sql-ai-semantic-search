# 🔍 SQL AI Semantic Search

[![Python](https://img.shields.io/badge/python-3.9+-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)

A high-performance semantic search implementation utilizing **PostgreSQL** as a vector database. This project demonstrates how to store high-dimensional text embeddings and perform cosine similarity searches using the `pgvector` extension and OpenAI's `text-embedding-3-small` model.


---

## 🚀 The Tech Stack

* **Database:** PostgreSQL with `pgvector` extension.
* **AI Integration:** OpenAI API (Embeddings).
* **Backend:** Python with `psycopg2` and `numpy`.
* **Infrastructure:** Dockerized database environment.

---

## ⚙️ How it Works

1. **Vectorization:** Raw text reviews are converted into 1536-dimensional vectors using OpenAI's embedding model.
2. **Storage:** Vectors are stored in a PostgreSQL table using the `VECTOR` data type.
3. **Semantic Retrieval:** User queries are embedded on the fly, and a **Cosine Distance** (`<=>`) calculation is performed in-database to find the most relevant contextually similar results.

---

## 🛠️ Setup & Installation

### 1. Database Requirements
Ensure you have a PostgreSQL instance running with the `pgvector` extension installed. If using Docker:
```bash
docker run --name pgvector-container -e POSTGRES_PASSWORD=your_password -p 5432:5432 -d ankane/pgvector
