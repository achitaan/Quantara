# Quantara Backend

## Overview
This is the backend for the Quantara project, built with Python, FastAPI, Chainlit, and LangChain. It provides an AI-powered chat workflow and integrates with PostgreSQL for persistent data storage.

## Requirements
- Python 3.10+
- PostgreSQL
- (Optional) Node.js and npm for the frontend

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-repo-url>
cd Quantara/backend
```

### 2. Create and Activate a Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

### 3. Install Python Dependencies
```
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the `backend` directory with the following content:
```
OPENAI_API_KEY=your_openai_api_key
CHAINLIT_AUTH_SECRET="secret"
DATABASE_URL=postgresql://root:1412@localhost:5432/quantara
```
Replace `your_openai_api_key` with your actual OpenAI API key.

### 5. Set Up PostgreSQL Database
1. **Install PostgreSQL** if you haven't already.
2. **Create the database and user:**
   ```sql
   CREATE DATABASE quantara;
   CREATE USER root WITH PASSWORD '';
   GRANT ALL PRIVILEGES ON DATABASE quantara TO root;
   ```
3. **Create required tables:**
   ```sql
   -- User table
   CREATE TABLE "User" (
     id UUID PRIMARY KEY,
     identifier VARCHAR(255) UNIQUE NOT NULL,
     metadata JSONB,
     "createdAt" TIMESTAMP DEFAULT NOW(),
     "updatedAt" TIMESTAMP DEFAULT NOW()
   );

   -- Thread table
   CREATE TABLE "Thread" (
     id UUID PRIMARY KEY,
     "userId" UUID REFERENCES "User"(id),
     "createdAt" TIMESTAMP DEFAULT NOW(),
     "deletedAt" TIMESTAMP
   );
   ```
4. **Grant permissions:**
   ```sql
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO root;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO root;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO root;
   ```

### 6. Run the Backend
```
chainlit run app.py
```

The backend will be available at [http://localhost:8000](http://localhost:8000).

## Troubleshooting
- If you see errors about missing columns, add them using `ALTER TABLE ... ADD COLUMN ...`.
- If you see permission errors, re-run the GRANT statements above.
- For translation warnings, US English (`en-US`) is used by default.

## License
MIT
