## Database Setup (PostgreSQL)

1. **Install PostgreSQL**  
   Download and install PostgreSQL from [https://www.postgresql.org/download/](https://www.postgresql.org/download/).

2. **Create the Database and User**  
   Open your SQL client (pgAdmin, psql, etc.) and run:

   ```sql
   CREATE DATABASE quantara;
   CREATE USER root WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE quantara TO root;
   ```

3. **Create the Tables**  
   Connect to the `quantara` database and run:

   ```sql
   -- Thread table (Chainlit expects only camelCase columns, no duplicates)
   CREATE TABLE "Thread" (
     id UUID PRIMARY KEY,
     "userId" UUID,
     "createdAt" TIMESTAMP,
     "deletedAt" TIMESTAMP
   );

   -- User table
   CREATE TABLE "User" (
     id UUID PRIMARY KEY,
     identifier VARCHAR(255) UNIQUE NOT NULL,
     metadata JSONB,
     "createdAt" TIMESTAMP,
     "updatedAt" TIMESTAMP
   );
   ```

   > **Note:**  
   > - Column names are case-sensitive and must match exactly what Chainlit expects.
   > - Do **not** include duplicate or snake_case columns (e.g., `created_at`, `uuid_id`).
   > - If you previously created the tables with extra columns, drop and recreate them as above.

4. **Set Permissions**  
   As a superuser, run:

   ```sql
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO root;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO root;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO root;
   ```

5. **Configure Environment Variables**  
   In your `.env` file, set:

   ```
   DATABASE_URL=postgresql://root:your_password@localhost:5432/quantara
   CHAINLIT_AUTH_SECRET=your_super_secret_key
   ```

6. **Troubleshooting**  
   - If you see errors about missing columns, add them using `ALTER TABLE ... ADD COLUMN ...`.
   - If you see permission errors, re-run the GRANT statements above.
   - Make sure column names and casing match exactly what your app expects.
   - If you get a `syntax error at or near ";"`, check for duplicate or incorrectly named columns and fix your schema.

---

This will ensure your database is ready for the app to run without schema or permission errors.``````