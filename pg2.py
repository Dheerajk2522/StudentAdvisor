import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
import glob
from tqdm import tqdm
import argparse
import streamlit as st

# Database connection parameters
DB_NAME = "studentAdv2"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# OpenAI API parameters
client = OpenAI(api_key=OPENAI_API_KEY)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# Function to generate embedding for a text
def generate_embedding(text):
    if not text or pd.isna(text):
        text = ""  # Handle None/NaN values
    return model.encode(str(text))

# Function to store query embedding
def store_query_embedding(query_text):
    """Store query embedding in the sentence_embeddings table"""
    try:
        embedding = generate_embedding(query_text)
        
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        
        cursor = conn.cursor()
        
        # Check if the query already exists
        cursor.execute("SELECT id FROM sentence_embeddings WHERE text = %s", (query_text,))
        result = cursor.fetchone()
        
        if result:
            # Update existing embedding
            cursor.execute(
                "UPDATE sentence_embeddings SET embedding = %s WHERE text = %s",
                (embedding.tolist(), query_text)
            )
        else:
            # Insert new embedding
            cursor.execute(
                "INSERT INTO sentence_embeddings (text, embedding) VALUES (%s, %s)",
                (query_text, embedding.tolist())
            )
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error storing query embedding: {e}")
        raise

# Function to search database with query
def search_database(query_text, similarity_threshold=0.3, max_results=15):
    """Search the database using the stored function"""
    try:
        # First store the query embedding
        store_query_embedding(query_text)
        
        # Now execute the search function
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM search_student_advising(:query, :threshold, :max_results)"),
                {"query": query_text, "threshold": similarity_threshold, "max_results": max_results}
            )
            
            # Convert rows to dictionaries with explicit column naming to handle results properly
            rows = []
            for row in result:
                rows.append({
                    'source_table': row[0],
                    'record_info': row[1],
                    'similarity': row[2]
                })
            
            return rows
            
    except Exception as e:
        print(f"Error searching database: {e}")
        raise

# Function to process query with OpenAI
def process_query_with_openai(user_query, search_results):
    """Process query with OpenAI using search results as context"""
    try:
        # Format search results as context
        context = json.dumps(search_results, indent=2)
        
        # Create system message with context
        system_message = f"""
        Act as a student advisor. You are an AI assistant for a student advising system.
        
        Always structure your response in the following format:
        
        1. Begin with "Student Advisory Response for [Student Name]"
        
        2. Student Profile section:
           - Name: [Full Name]
           - Career: [Academic Career]
           - Institution: [Institution Name]
           - Current Term: [Current Term]
           - Current Courses Taken: [List of current courses with codes, titles, and GPA if available]
        
        3. Step 1: Identifying the Student's Study Plan
           - Identify the program the student is in based on courses and career
           - Reference relevant study plans and program requirements
           
        4. Step 2: Understanding Prerequisites and Progression
           - List possible next courses with their codes and titles
           - Explain prerequisites for each recommended course
           - Provide justification for why each course is appropriate
        
        5. Step 3: Checking Term Availability
           - List when and where each recommended course is available
           - Note any scheduling constraints or conflicts
        
        6. Final Course Recommendation for Next Semester
           - Provide a bulleted list of 3-5 specific course recommendations with codes, titles, and schedule
        
        7. Conclusion
           - Summarize the recommendations and how they align with the student's academic progress
           - Explain the benefits of this course selection for their program
        
        Base all recommendations on the student's current courses, GPA, program requirements, prerequisites, course availability, and appropriate study plan progression. Always provide specific course codes and titles.
        
        If the query is about a specific student, analyze their information comprehensively.
        If the query is not related to study plans and courses, provide general academic advice but maintain a similar structured format.
        
        Use the following search results as context to answer the user's query:
        
        Search Results:
        {context}
        """
    
        # Call OpenAI to generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error processing query with OpenAI: {e}")
        raise

def insert_data_with_embeddings(data_dict):
    """Insert data with embeddings into the database"""
    # Connect to PostgreSQL using psycopg2 for better control over array insertion
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    
    try:
        cursor = conn.cursor()
        
        # Process each table
        for table_name, df in data_dict.items():
            print(f"Processing data for table: {table_name}")
            
            if len(df) == 0:
                print(f"  No data to insert for {table_name}")
                continue
            
            # Generate text for embeddings based on table
            if table_name == "students_enrollment":
                df['text_for_embedding'] = df.apply(
                    lambda row: ' '.join(str(val) for val in [
                        row.get('name_display', ''),
                        row.get('acad_career', ''),
                        row.get('institution', ''),
                        row.get('acad_prog', ''),
                        row.get('descr', ''),
                        row.get('course_title_long', ''),
                        row.get('subject', ''),
                        row.get('catalog_nbr', '')
                    ] if val and not pd.isna(val)),
                    axis=1
                )
                
                # Generate embeddings individually to avoid conflicts
                print(f"  Generating embeddings for {len(df)} rows...")
                for index, row in tqdm(df.iterrows(), total=len(df)):
                    try:
                        # Generate embedding for this row
                        embedding = generate_embedding(row['text_for_embedding'])
                        
                        # Insert data for this single row with direct cursor execution
                        insert_query = """
                        INSERT INTO students_enrollment (
                            emplid, name_display, acad_career, institution, strm,
                            class_nbr, unt_taken, acad_prog, descr, crse_id,
                            crse_grade_off, course_title_long, cum_gpa, subject,
                            catalog_nbr, acad_org, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (emplid, class_nbr, crse_id) DO UPDATE SET
                            name_display = EXCLUDED.name_display,
                            acad_career = EXCLUDED.acad_career,
                            institution = EXCLUDED.institution,
                            strm = EXCLUDED.strm,
                            unt_taken = EXCLUDED.unt_taken,
                            acad_prog = EXCLUDED.acad_prog,
                            descr = EXCLUDED.descr,
                            crse_grade_off = EXCLUDED.crse_grade_off,
                            course_title_long = EXCLUDED.course_title_long,
                            cum_gpa = EXCLUDED.cum_gpa,
                            subject = EXCLUDED.subject,
                            catalog_nbr = EXCLUDED.catalog_nbr,
                            acad_org = EXCLUDED.acad_org,
                            embedding = EXCLUDED.embedding
                        """
                        
                        cursor.execute(insert_query, (
                            row.get('emplid'),
                            row.get('name_display'),
                            row.get('acad_career'),
                            row.get('institution'),
                            row.get('strm'),
                            row.get('class_nbr'),
                            row.get('unt_taken'),
                            row.get('acad_prog'),
                            row.get('descr'),
                            row.get('crse_id'),
                            row.get('crse_grade_off'),
                            row.get('course_title_long'),
                            row.get('cum_gpa'),
                            row.get('subject'),
                            row.get('catalog_nbr'),
                            row.get('acad_org'),
                            embedding.tolist()
                        ))
                        
                        # Commit every 100 rows to avoid long transactions
                        if index % 100 == 0:
                            conn.commit()
                            
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        # Continue with next row
                
                # Final commit for students_enrollment
                conn.commit()
                print(f"  Data inserted successfully for {table_name}")
                
            elif table_name == "acad_prog":
                df['text_for_embedding'] = df.apply(
                    lambda row: ' '.join(str(val) for val in [
                        row.get('acad_career', ''),
                        row.get('acad_prog', ''),
                        row.get('prog_status', ''),
                        row.get('prog_action', ''),
                        row.get('campus', '')
                    ] if val and not pd.isna(val)),
                    axis=1
                )
                
                # Generate embeddings individually to avoid conflicts
                print(f"  Generating embeddings for {len(df)} rows...")
                for index, row in tqdm(df.iterrows(), total=len(df)):
                    try:
                        # Generate embedding for this row
                        embedding = generate_embedding(row['text_for_embedding'])
                        
                        # Insert data for this single row with direct cursor execution
                        insert_query = """
                        INSERT INTO acad_prog (
                            emplid, acad_career, acad_prog, prog_status,
                            prog_action, admit_term, campus, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (emplid, acad_prog) DO UPDATE SET
                            acad_career = EXCLUDED.acad_career,
                            prog_status = EXCLUDED.prog_status,
                            prog_action = EXCLUDED.prog_action,
                            admit_term = EXCLUDED.admit_term,
                            campus = EXCLUDED.campus,
                            embedding = EXCLUDED.embedding
                        """
                        
                        cursor.execute(insert_query, (
                            row.get('emplid'),
                            row.get('acad_career'),
                            row.get('acad_prog'),
                            row.get('prog_status'),
                            row.get('prog_action'),
                            row.get('admit_term'),
                            row.get('campus'),
                            embedding.tolist()
                        ))
                        
                        # Commit every 100 rows to avoid long transactions
                        if index % 100 == 0:
                            conn.commit()
                            
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        # Continue with next row
                
                # Final commit for acad_prog
                conn.commit()
                print(f"  Data inserted successfully for {table_name}")
                    
            elif table_name == "crse_catalog":
                df['text_for_embedding'] = df.apply(
                    lambda row: ' '.join(str(val) for val in [
                        row.get('descr', ''),
                        row.get('long_title', ''),
                        row.get('subject', ''),
                        row.get('catalog', ''),
                        row.get('acad_group', ''),
                        row.get('acad_org', ''),
                        row.get('career', ''),
                        row.get('institution', '')
                    ] if val and not pd.isna(val)),
                    axis=1
                )
                
                # Generate embeddings individually to avoid conflicts
                print(f"  Generating embeddings for {len(df)} rows...")
                for index, row in tqdm(df.iterrows(), total=len(df)):
                    try:
                        # Generate embedding for this row
                        embedding = generate_embedding(row['text_for_embedding'])
                        
                        # Insert data for this single row with direct cursor execution
                        insert_query = """
                        INSERT INTO crse_catalog (
                            course_id, eff_date, status, descr,
                            min_units, max_units, long_title, offer_nbr,
                            acad_group, subject, catalog, campus,
                            acad_org, career, institution, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (course_id) DO UPDATE SET
                            eff_date = EXCLUDED.eff_date,
                            status = EXCLUDED.status,
                            descr = EXCLUDED.descr,
                            min_units = EXCLUDED.min_units,
                            max_units = EXCLUDED.max_units,
                            long_title = EXCLUDED.long_title,
                            offer_nbr = EXCLUDED.offer_nbr,
                            acad_group = EXCLUDED.acad_group,
                            subject = EXCLUDED.subject,
                            catalog = EXCLUDED.catalog,
                            campus = EXCLUDED.campus,
                            acad_org = EXCLUDED.acad_org,
                            career = EXCLUDED.career,
                            institution = EXCLUDED.institution,
                            embedding = EXCLUDED.embedding
                        """
                        
                        cursor.execute(insert_query, (
                            row.get('course_id'),
                            row.get('eff_date'),
                            row.get('status'),
                            row.get('descr'),
                            row.get('min_units'),
                            row.get('max_units'),
                            row.get('long_title'),
                            row.get('offer_nbr'),
                            row.get('acad_group'),
                            row.get('subject'),
                            row.get('catalog'),
                            row.get('campus'),
                            row.get('acad_org'),
                            row.get('career'),
                            row.get('institution'),
                            embedding.tolist()
                        ))
                        
                        # Commit every 100 rows to avoid long transactions
                        if index % 100 == 0:
                            conn.commit()
                            
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        # Continue with next row
                
                # Final commit for crse_catalog
                conn.commit()
                print(f"  Data inserted successfully for {table_name}")
                    
            elif table_name == "study_plan":
                df['text_for_embedding'] = df.apply(
                    lambda row: ' '.join(str(val) for val in [
                        row.get('code', ''),
                        row.get('title', ''),
                        row.get('prerequisite', ''),
                        str(row.get('year', '')),
                        row.get('semester', '')
                    ] if val and not pd.isna(val)),
                    axis=1
                )
                
                # Generate embeddings individually to avoid conflicts
                print(f"  Generating embeddings for {len(df)} rows...")
                for index, row in tqdm(df.iterrows(), total=len(df)):
                    # Use a new connection for each row to avoid transaction issues
                    row_conn = None
                    row_cursor = None
                    try:
                        # Generate embedding for this row
                        embedding = generate_embedding(row['text_for_embedding'])
                        
                        # Handle credits field - could be 'cr_' or 'credits'
                        credits_value = row.get('credits', row.get('cr_'))
                        if credits_value is not None:
                            try:
                                # Convert to float first, then to string to ensure proper decimal handling
                                credits_value = float(str(credits_value).replace(',', '.'))
                                if credits_value > 1000:  # Sanity check
                                    credits_value = None
                            except (ValueError, TypeError):
                                credits_value = None
                        
                        # Handle year field
                        year_value = row.get('year')
                        if year_value is not None:
                            try:
                                # Ensure year is a valid integer
                                year_value = int(float(str(year_value)))
                                if year_value < 1 or year_value > 10:  # Assuming study plan years are 1-10
                                    year_value = None
                            except (ValueError, TypeError):
                                year_value = None
                        
                        # Create a new connection for this row
                        row_conn = psycopg2.connect(
                            dbname=DB_NAME,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            host=DB_HOST,
                            port=DB_PORT
                        )
                        row_cursor = row_conn.cursor()
                        
                        # Insert data for this single row with direct cursor execution
                        insert_query = """
                        INSERT INTO study_plan (
                            code, title, credits, prerequisite,
                            year, semester, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (code) DO UPDATE SET
                            title = EXCLUDED.title,
                            credits = EXCLUDED.credits,
                            prerequisite = EXCLUDED.prerequisite,
                            year = EXCLUDED.year,
                            semester = EXCLUDED.semester,
                            embedding = EXCLUDED.embedding
                        """
                        
                        row_cursor.execute(insert_query, (
                            row.get('code'),
                            row.get('title'),
                            credits_value,
                            row.get('prerequisite'),
                            year_value,
                            row.get('semester'),
                            embedding.tolist()
                        ))
                        
                        row_conn.commit()
                            
                    except Exception as e:
                        if row_conn:
                            row_conn.rollback()
                        print(f"Error processing row {index} for study_plan: {e}")
                    finally:
                        if row_cursor:
                            row_cursor.close()
                        if row_conn:
                            row_conn.close()
                
                print(f"  Data inserted successfully for {table_name}")
            
            elif table_name == "term_history" or table_name == "class_schedule" or table_name == "time_table":
                # These tables could also benefit from row-by-row processing
                df['text_for_embedding'] = df.apply(
                    lambda row: ' '.join(str(val) for val in row.values if val and not pd.isna(val)),
                    axis=1
                )
                
                # Process each row individually
                print(f"  Generating embeddings for {len(df)} rows...")
                for index, row in tqdm(df.iterrows(), total=len(df)):
                    # Use a new connection for each row to avoid transaction issues
                    row_conn = None
                    row_cursor = None
                    try:
                        # Generate embedding
                        embedding = generate_embedding(row['text_for_embedding'])
                        
                        # Create column names and values for the query
                        cols = [col for col in row.index if col != 'text_for_embedding']
                        cols.append('embedding')
                        
                        # Create placeholders for SQL query
                        placeholders = ', '.join(['%s'] * len(cols))
                        
                        # Prepare values with proper data type handling
                        values = []
                        for col in cols[:-1]:  # All columns except embedding
                            val = row.get(col)
                            
                            # Handle special data types for class_schedule
                            if table_name == "class_schedule" and col in ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']:
                                # Convert various boolean representations to proper boolean
                                if isinstance(val, bool):
                                    values.append(val)
                                elif isinstance(val, (int, float)):
                                    values.append(bool(val))
                                elif isinstance(val, str):
                                    values.append(val.lower() in ['true', 'yes', '1', 'y', 't'])
                                else:
                                    values.append(False)
                            elif col in ['mtg_start', 'mtg_end'] and val is not None:
                                # Handle time fields
                                if pd.isna(val) or val == 'NaN' or val == 'nan':
                                    # Handle NaN values as NULL
                                    values.append(None)
                                elif isinstance(val, str):
                                    # Try to convert string time format to a PostgreSQL-compatible format
                                    try:
                                        # Get just the time portion if it includes AM/PM
                                        time_parts = val.replace('AM', '').replace('PM', '').strip().split(':')
                                        if len(time_parts) >= 2:
                                            # If PM and not 12, add 12 hours
                                            hour = int(time_parts[0])
                                            if 'PM' in val.upper() and hour < 12:
                                                hour += 12
                                            # Format as HH:MM:SS
                                            time_val = f"{hour:02d}:{time_parts[1]:0>2}:00"
                                            values.append(time_val)
                                        else:
                                            # Can't parse properly, use NULL
                                            values.append(None)
                                    except:
                                        # If any parsing error, use NULL
                                        values.append(None)
                                else:
                                    # Not string and not NaN, pass as is
                                    values.append(None)
                            elif col in ['start_date', 'end_date', 'eff_date'] and val is not None:
                                # Handle date fields
                                if pd.isna(val) or val == 'NaN' or val == 'nan':
                                    values.append(None)
                                elif isinstance(val, str):
                                    # Try to convert string date format (e.g., 24-02-2025) to SQL date format
                                    try:
                                        # Handle common date formats
                                        if '-' in val:
                                            parts = val.split('-')
                                            if len(parts) == 3:
                                                # Assuming DD-MM-YYYY format
                                                day, month, year = parts
                                                # Convert to YYYY-MM-DD for SQL
                                                values.append(f"{year}-{month}-{day}")
                                            else:
                                                values.append(val)  # Use as is if format unclear
                                        else:
                                            values.append(val)  # Use as is if format unclear
                                    except:
                                        values.append(None)
                                else:
                                    # Handle pandas Timestamp or other date objects
                                    try:
                                        date_str = str(val).split()[0]  # Get just the date part
                                        values.append(date_str)
                                    except:
                                        values.append(None)
                            else:
                                values.append(val)
                        
                        # Add embedding
                        values.append(embedding.tolist())
                        
                        # Create a new connection and cursor for this row
                        row_conn = psycopg2.connect(
                            dbname=DB_NAME,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            host=DB_HOST,
                            port=DB_PORT
                        )
                        row_cursor = row_conn.cursor()
                        
                        # Set primary key constraint handling based on table
                        if table_name == "class_schedule":
                            conflict_clause = "ON CONFLICT (class_nbr, term) DO NOTHING"
                        elif table_name == "term_history":
                            conflict_clause = "ON CONFLICT (emplid, strm) DO NOTHING"
                        elif table_name == "time_table":
                            conflict_clause = "ON CONFLICT (id, class_nbr, term) DO NOTHING"
                        else:
                            conflict_clause = "ON CONFLICT DO NOTHING"
                        
                        # Create INSERT query
                        insert_query = f"""
                        INSERT INTO {table_name} ({', '.join(cols)})
                        VALUES ({placeholders})
                        {conflict_clause}
                        """
                        
                        # Execute and commit for this single row
                        row_cursor.execute(insert_query, values)
                        row_conn.commit()
                        
                    except Exception as e:
                        # If error occurs, rollback this row's transaction
                        if row_conn:
                            row_conn.rollback()
                        print(f"Error processing row {index} for {table_name}: {e}")
                    finally:
                        # Always close the per-row connection
                        if row_cursor:
                            row_cursor.close()
                        if row_conn:
                            row_conn.close()
                
                print(f"  Data inserted successfully for {table_name}")
            
            else:
                print(f"  Skipping unknown table: {table_name}")
                continue
            
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Function to set up pgvector extension
def setup_pgvector():
    """Set up pgvector extension in PostgreSQL"""
    try:
        with engine.connect() as conn:
            # Create extension if it doesn't exist
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("pgvector extension set up successfully")
    except SQLAlchemyError as e:
        print(f"Error setting up pgvector extension: {e}")
        raise

# Function to create vector indexes on tables
def create_vector_indexes():
    """Create vector indexes for faster similarity search"""
    index_definitions = [
        {"table": "students_enrollment", "query": "CREATE INDEX IF NOT EXISTS idx_students_embedding ON students_enrollment USING ivfflat (embedding vector_cosine_ops);"},
        {"table": "acad_prog", "query": "CREATE INDEX IF NOT EXISTS idx_acad_prog_embedding ON acad_prog USING ivfflat (embedding vector_cosine_ops);"},
        {"table": "crse_catalog", "query": "CREATE INDEX IF NOT EXISTS idx_crse_catalog_embedding ON crse_catalog USING ivfflat (embedding vector_cosine_ops);"},
        {"table": "study_plan", "query": "CREATE INDEX IF NOT EXISTS idx_study_plan_embedding ON study_plan USING ivfflat (embedding vector_cosine_ops);"},
        {"table": "sentence_embeddings", "query": "CREATE INDEX IF NOT EXISTS idx_sentence_embeddings ON sentence_embeddings USING ivfflat (embedding vector_cosine_ops);"}
    ]
    
    success_count = 0
    
    with engine.connect() as conn:
        # Check which tables exist first
        existing_tables = []
        try:
            result = conn.execute(text("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            existing_tables = [row[0] for row in result]
        except SQLAlchemyError as e:
            print(f"Warning: Could not check existing tables: {e}")
        
        # Create indexes only for tables that exist
        for index_def in index_definitions:
            try:
                table_name = index_def["table"]
                if table_name in existing_tables:
                    conn.execute(text(index_def["query"]))
                    conn.commit()
                    success_count += 1
                else:
                    print(f"Skipping index creation for {table_name} - table does not exist yet")
            except SQLAlchemyError as e:
                print(f"Error creating index for {index_def['table']}: {e}")
    
    print(f"Vector indexes created successfully: {success_count}/{len(index_definitions)}")

# Function to load data from Excel files
def load_excel_data(data_folder):
    file_mapping = {
        "20_STUDENTS_ENROLLEMENT.xlsx": {
            "table": "students_enrollment",
            "columns": [
                'EMPLID', 'NAME_DISPLAY', 'ACAD_CAREER', 'INSTITUTION', 'STRM', 
                'CLASS_NBR', 'UNT_TAKEN', 'ACAD_PROG', 'DESCR', 'CRSE_ID', 
                'CRSE_GRADE_OFF', 'COURSE_TITLE_LONG', 'CUM_GPA', 'SUBJECT', 
                'CATALOG_NBR', 'ACAD_ORG'
            ]
        },
        "ACAD_PROG.xlsx": {
            "table": "acad_prog",
            "columns": [
                'EMPLID', 'ACAD_CAREER', 'ACAD_PROG', 'PROG_STATUS', 'PROG_ACTION', 
                'ADMIT_TERM', 'CAMPUS'
            ]
        },
        "TERM_HISTORY.xlsx": {
            "table": "term_history",
            "columns": [
                'EMPLID', 'TOT_TAKEN_PRGRSS', 'TOT_PASSD_PRGRSS', 'ACAD_CAREER', 
                'INSTITUTION', 'ACAD_PROG_PRIMARY', 'STRM', 'TERM_GPA'
            ]
        },
        "CRSE_CATALOG_85.xlsx": {
            "table": "crse_catalog",
            "columns": [
                'Course ID', 'Eff Date', 'Status', 'Descr', 'Min Units', 'Max Units', 
                'Long Title', 'Offer Nbr', 'Acad Group', 'Subject', 'Catalog', 
                'Campus', 'Acad Org', 'Career', 'Institution'
            ]
        },
        "CLASS_SCHEDULE_4764.xlsx": {
            "table": "class_schedule",
            "columns": [
                'Course ID', 'Term', 'Offer Nbr', 'Acad Group', 'Subject', 'Catalog', 
                'Descr', 'Class Nbr', 'Cap Enrl', 'Tot Enrl', 'Acad Org', 'Campus', 
                'Section', 'ID', 'Role', 'Facil ID', 'Mtg Start', 'Mtg End', 'Mon', 
                'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'Display Name'
            ]
        },
        "TIME_TABLE.xlsx": {
            "table": "time_table",
            "columns": [
                'ID', 'Student Name', 'Institution', 'Term', 'Class Nbr', 'Session', 
                'Section', 'Course ID', 'Mtg Start', 'Mtg End', 'Start Date', 'End Date', 
                'Pat', 'Descr', 'Course Descr', 'Instructor ID', 'Role', 'Instructor Name'
            ]
        },
        "BArchitectureStudyPlan2022.xlsx": {
            "table": "study_plan",
            "columns": [
                'Code', 'Title', 'Cr.', 'Prerequisite', 'Year', 'Semester'
            ]
        }
    }
    
    loaded_data = {}
    for file_pattern, config in file_mapping.items():
        matching_files = glob.glob(os.path.join(data_folder, file_pattern))
        if not matching_files:
            print(f"Warning: No files found matching pattern: {file_pattern}")
            continue
            
        for file_path in matching_files:
            try:
                print(f"Loading data from: {file_path}")
                df = pd.read_excel(file_path)
                
                # Normalize column names
                df.columns = [col.lower().replace(' ', '_').replace('.', '_') for col in df.columns]
                
                # Handle specific column renames for study plan
                if config["table"] == "study_plan":
                    if 'cr_' in df.columns:
                        df = df.rename(columns={'cr_': 'credits'})
                
                # Store the loaded data
                loaded_data[config["table"]] = df
                print(f"  Loaded {len(df)} rows for {config['table']}")
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")
    
    return loaded_data

# Function to create tables
def create_tables():
    drop_statements = [
        "DROP TABLE IF EXISTS students_enrollment CASCADE;",
        "DROP TABLE IF EXISTS acad_prog CASCADE;",
        "DROP TABLE IF EXISTS term_history CASCADE;",
        "DROP TABLE IF EXISTS crse_catalog CASCADE;",
        "DROP TABLE IF EXISTS class_schedule CASCADE;",
        "DROP TABLE IF EXISTS time_table CASCADE;",
        "DROP TABLE IF EXISTS study_plan CASCADE;",
        "DROP TABLE IF EXISTS sentence_embeddings CASCADE;"
    ]
    
    table_definitions = [
        """
        CREATE TABLE IF NOT EXISTS students_enrollment (
          emplid VARCHAR(20),
          name_display TEXT,
          acad_career TEXT,
          institution TEXT,
          strm TEXT,
          class_nbr INTEGER,
          unt_taken NUMERIC(5,2),
          acad_prog TEXT,
          descr TEXT,
          crse_id VARCHAR(20),
          crse_grade_off TEXT,
          course_title_long TEXT,
          cum_gpa NUMERIC(5,2),
          subject TEXT,
          catalog_nbr TEXT,
          acad_org TEXT,
          embedding vector(384),
          PRIMARY KEY (emplid, class_nbr, crse_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS acad_prog (
          emplid VARCHAR(20),
          acad_career TEXT,
          acad_prog TEXT,
          prog_status TEXT,
          prog_action TEXT,
          admit_term TEXT,
          campus TEXT,
          embedding vector(384),
          PRIMARY KEY (emplid, acad_prog)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS term_history (
          emplid VARCHAR(20),
          tot_taken_prgrss NUMERIC(5,2),
          tot_passd_prgrss NUMERIC(5,2),
          acad_career TEXT,
          institution TEXT,
          acad_prog_primary TEXT,
          strm TEXT,
          term_gpa NUMERIC(5,2),
          embedding vector(384),
          PRIMARY KEY (emplid, strm)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS crse_catalog (
          course_id VARCHAR(20) PRIMARY KEY,
          eff_date DATE,
          status TEXT,
          descr TEXT,
          min_units NUMERIC(5,2),
          max_units NUMERIC(5,2),
          long_title TEXT,
          offer_nbr INTEGER,
          acad_group TEXT,
          subject TEXT,
          catalog TEXT,
          campus TEXT,
          acad_org TEXT,
          career TEXT,
          institution TEXT,
          embedding vector(384)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS class_schedule (
          course_id VARCHAR(20),
          term TEXT,
          offer_nbr INTEGER,
          acad_group TEXT,
          subject TEXT,
          catalog TEXT,
          descr TEXT,
          class_nbr INTEGER,
          cap_enrl INTEGER,
          tot_enrl INTEGER,
          acad_org TEXT,
          campus TEXT,
          section TEXT,
          id VARCHAR(20),
          role TEXT,
          facil_id TEXT,
          mtg_start TIME,
          mtg_end TIME,
          mon BOOLEAN,
          tues BOOLEAN,
          wed BOOLEAN,
          thurs BOOLEAN,
          fri BOOLEAN,
          sat BOOLEAN,
          sun BOOLEAN,
          display_name TEXT,
          embedding vector(384),
          PRIMARY KEY (class_nbr, term)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS time_table (
          id VARCHAR(20),
          student_name TEXT,
          institution TEXT,
          term TEXT,
          class_nbr INTEGER,
          session TEXT,
          section TEXT,
          course_id VARCHAR(20),
          mtg_start TIME,
          mtg_end TIME,
          start_date DATE,
          end_date DATE,
          pat TEXT,
          descr TEXT,
          course_descr TEXT,
          instructor_id VARCHAR(20),
          role TEXT,
          instructor_name TEXT,
          embedding vector(384),
          PRIMARY KEY (id, class_nbr, term)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS study_plan (
          code TEXT PRIMARY KEY,
          title TEXT,
          credits NUMERIC(5,2),
          prerequisite TEXT,
          year INTEGER,
          semester TEXT,
          embedding vector(384)
        );
        """
    ]
    
    try:
        with engine.connect() as conn:
            for drop_statement in drop_statements:
                conn.execute(text(drop_statement))
                conn.commit()
            for table_definition in table_definitions:
                conn.execute(text(table_definition))
                conn.commit()
            print("Tables created successfully")
    except SQLAlchemyError as e:
        print(f"Error creating tables: {e}")
        raise

# Function to create the search function in PostgreSQL
def create_search_function():
    # Create a table to store embeddings for queries
    create_embeddings_table = """
    CREATE TABLE IF NOT EXISTS sentence_embeddings (
        id SERIAL PRIMARY KEY,
        text TEXT UNIQUE,
        embedding vector(384)
    );
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(create_embeddings_table))
            conn.commit()
            print("Embeddings table created successfully")
    except SQLAlchemyError as e:
        print(f"Error creating embeddings table: {e}")
        raise
    
    search_function = """
    CREATE OR REPLACE FUNCTION search_student_advising(
        query_text TEXT, 
        similarity_threshold FLOAT DEFAULT 0.3, 
        max_results INT DEFAULT 15
    )
    RETURNS TABLE (
        source_table TEXT,
        record_info JSONB,
        similarity FLOAT
    ) AS $$
    DECLARE
        query_embedding vector;
        student_names TEXT[];
        student_ids TEXT[];
        has_student BOOLEAN := false;
        student_id TEXT;
        var_student_name TEXT;  -- Renamed to avoid ambiguity
    BEGIN
        -- Get embedding for the query
        SELECT embedding INTO query_embedding FROM sentence_embeddings WHERE text = query_text;
        
        -- Extract potential student names/IDs from the query
        -- First try to find student IDs (numeric patterns)
        student_ids := ARRAY(
            SELECT DISTINCT se.emplid FROM students_enrollment se 
            WHERE query_text ILIKE '%' || se.emplid || '%'
        );
        
        -- Next try to find student names
        student_names := ARRAY(
            SELECT DISTINCT se.name_display FROM students_enrollment se 
            WHERE position(lower(se.name_display) in lower(query_text)) > 0
        );
        
        -- Set flag if we found any students
        has_student := array_length(student_ids, 1) > 0 OR array_length(student_names, 1) > 0;
        
        -- If there's a specific student, prioritize their information
        IF has_student THEN
            -- Get the first student ID if available
            IF array_length(student_ids, 1) > 0 THEN
                student_id := student_ids[1];
            -- Otherwise use the name to find the ID
            ELSIF array_length(student_names, 1) > 0 THEN
                var_student_name := student_names[1];
                SELECT emplid INTO student_id FROM students_enrollment 
                WHERE name_display = var_student_name LIMIT 1;
            END IF;
            
            -- Return all information related to this student
            IF student_id IS NOT NULL THEN
                -- First return student enrollment info
                RETURN QUERY
                SELECT 
                    'students_enrollment' AS source_table,
                    jsonb_build_object(
                        'emplid', se.emplid,
                        'name', se.name_display,
                        'program', se.acad_prog,
                        'career', se.acad_career,
                        'gpa', se.cum_gpa,
                        'course', se.course_title_long,
                        'course_id', se.crse_id,
                        'class_nbr', se.class_nbr,
                        'subject', se.subject,
                        'catalog_nbr', se.catalog_nbr,
                        'institution', se.institution,
                        'term', se.strm,
                        'grade', se.crse_grade_off
                    ) AS record_info,
                    1.0::FLOAT8 AS similarity
                FROM 
                    students_enrollment se
                WHERE 
                    se.emplid = student_id
                LIMIT 10;
                
                -- Also return academic program info
                RETURN QUERY
                SELECT 
                    'acad_prog' AS source_table,
                    jsonb_build_object(
                        'emplid', ap.emplid,
                        'program', ap.acad_prog,
                        'career', ap.acad_career,
                        'status', ap.prog_status,
                        'action', ap.prog_action,
                        'admit_term', ap.admit_term,
                        'campus', ap.campus
                    ) AS record_info,
                    1.0::FLOAT8 AS similarity
                FROM 
                    acad_prog ap
                WHERE 
                    ap.emplid = student_id;
                    
                -- Return term history
                RETURN QUERY
                SELECT 
                    'term_history' AS source_table,
                    jsonb_build_object(
                        'emplid', th.emplid,
                        'term', th.strm,
                        'gpa', th.term_gpa,
                        'career', th.acad_career,
                        'institution', th.institution,
                        'program', th.acad_prog_primary,
                        'units_taken', th.tot_taken_prgrss,
                        'units_passed', th.tot_passd_prgrss
                    ) AS record_info,
                    1.0::FLOAT8 AS similarity
                FROM 
                    term_history th
                WHERE 
                    th.emplid = student_id;
                    
                -- Return time table for the student - Add null check for var_student_name
                IF var_student_name IS NOT NULL THEN
                    RETURN QUERY
                    SELECT 
                        'time_table' AS source_table,
                        jsonb_build_object(
                            'id', tt.id,
                            'student_name', tt.student_name,
                            'term', tt.term,
                            'course_id', tt.course_id,
                            'class_nbr', tt.class_nbr,
                            'descr', tt.descr,
                            'course_descr', tt.course_descr,
                            'start_time', tt.mtg_start,
                            'end_time', tt.mtg_end,
                            'start_date', tt.start_date,
                            'end_date', tt.end_date,
                            'instructor', tt.instructor_name
                        ) AS record_info,
                        1.0::FLOAT8 AS similarity
                    FROM 
                        time_table tt
                    WHERE 
                        tt.id = student_id OR tt.student_name ILIKE '%' || var_student_name || '%';
                ELSE
                    -- If no student name, just search by ID
                    RETURN QUERY
                    SELECT 
                        'time_table' AS source_table,
                        jsonb_build_object(
                            'id', tt.id,
                            'student_name', tt.student_name,
                            'term', tt.term,
                            'course_id', tt.course_id,
                            'class_nbr', tt.class_nbr,
                            'descr', tt.descr,
                            'course_descr', tt.course_descr,
                            'start_time', tt.mtg_start,
                            'end_time', tt.mtg_end,
                            'start_date', tt.start_date,
                            'end_date', tt.end_date,
                            'instructor', tt.instructor_name
                        ) AS record_info,
                        1.0::FLOAT8 AS similarity
                    FROM 
                        time_table tt
                    WHERE 
                        tt.id = student_id;
                END IF;
            END IF;
        END IF;
        
        -- Now perform semantic search on all tables
        RETURN QUERY
        SELECT * FROM (
            -- Search students with vector similarity
            SELECT 
                'students_enrollment' AS source_table,
                jsonb_build_object(
                    'emplid', se.emplid,
                    'name', se.name_display,
                    'program', se.acad_prog,
                    'career', se.acad_career,
                    'gpa', se.cum_gpa,
                    'course', se.course_title_long,
                    'course_id', se.crse_id,
                    'class_nbr', se.class_nbr,
                    'subject', se.subject,
                    'catalog_nbr', se.catalog_nbr,
                    'institution', se.institution,
                    'term', se.strm,
                    'grade', se.crse_grade_off
                ) AS record_info,
                (1 - (se.embedding <=> query_embedding))::FLOAT8 AS similarity
            FROM 
                students_enrollment se
            WHERE 
                1 - (se.embedding <=> query_embedding) > similarity_threshold
            
            UNION ALL
            
            -- Search course catalog with vector similarity
            SELECT 
                'crse_catalog' AS source_table,
                jsonb_build_object(
                    'course_id', cc.course_id,
                    'title', cc.long_title,
                    'description', cc.descr,
                    'subject', cc.subject,
                    'catalog', cc.catalog,
                    'units_min', cc.min_units,
                    'units_max', cc.max_units,
                    'acad_group', cc.acad_group,
                    'acad_org', cc.acad_org,
                    'career', cc.career,
                    'institution', cc.institution,
                    'campus', cc.campus,
                    'status', cc.status,
                    'eff_date', cc.eff_date
                ) AS record_info,
                (1 - (cc.embedding <=> query_embedding))::FLOAT8 AS similarity
            FROM 
                crse_catalog cc
            WHERE 
                1 - (cc.embedding <=> query_embedding) > similarity_threshold
            
            UNION ALL
            
            -- Search study plan with vector similarity
            SELECT 
                'study_plan' AS source_table,
                jsonb_build_object(
                    'code', sp.code,
                    'title', sp.title,
                    'credits', sp.credits,
                    'prerequisite', sp.prerequisite,
                    'year', sp.year,
                    'semester', sp.semester
                ) AS record_info,
                (1 - (sp.embedding <=> query_embedding))::FLOAT8 AS similarity
            FROM 
                study_plan sp
            WHERE 
                1 - (sp.embedding <=> query_embedding) > similarity_threshold
            
            UNION ALL
            
            -- Search academic programs with vector similarity
            SELECT 
                'acad_prog' AS source_table,
                jsonb_build_object(
                    'emplid', ap.emplid,
                    'program', ap.acad_prog,
                    'career', ap.acad_career,
                    'status', ap.prog_status,
                    'action', ap.prog_action,
                    'admit_term', ap.admit_term,
                    'campus', ap.campus
                ) AS record_info,
                (1 - (ap.embedding <=> query_embedding))::FLOAT8 AS similarity
            FROM 
                acad_prog ap
            WHERE 
                1 - (ap.embedding <=> query_embedding) > similarity_threshold
            
            UNION ALL
            
            -- Also search class schedule for relevant courses
            SELECT 
                'class_schedule' AS source_table,
                jsonb_build_object(
                    'course_id', cs.course_id,
                    'term', cs.term,
                    'subject', cs.subject,
                    'catalog', cs.catalog,
                    'description', cs.descr,
                    'class_nbr', cs.class_nbr,
                    'section', cs.section,
                    'start_time', cs.mtg_start,
                    'end_time', cs.mtg_end,
                    'days', jsonb_build_object(
                        'mon', cs.mon,
                        'tue', cs.tues,
                        'wed', cs.wed,
                        'thu', cs.thurs,
                        'fri', cs.fri,
                        'sat', cs.sat,
                        'sun', cs.sun
                    ),
                    'capacity', cs.cap_enrl,
                    'enrollment', cs.tot_enrl,
                    'instructor', cs.display_name
                ) AS record_info,
                0.9::FLOAT8 AS similarity
            FROM 
                class_schedule cs
            WHERE
                cs.subject ILIKE ANY(
                    array(SELECT DISTINCT cc.subject FROM crse_catalog cc 
                          WHERE 1 - (cc.embedding <=> query_embedding) > similarity_threshold)
                )
        ) AS combined_results
        ORDER BY 
            similarity DESC
        LIMIT 
            max_results;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(search_function))
            conn.commit()
            print("Search function created successfully")
    except SQLAlchemyError as e:
        print(f"Error creating search function: {e}")
        raise

# Main function to setup and run the system
def main(skip_loading=False, skip_tables=False, interactive=False):
    # Path to the directory containing Excel files
    data_folder = "./stu_excel"  # Updated to the correct folder
    
    if not skip_tables:
        # Setup pgvector extension
        print("Setting up pgvector extension...")
        setup_pgvector()
        
        # Create tables
        print("Creating tables...")
        create_tables()
        
        # Create search function FIRST (which creates the sentence_embeddings table)
        print("Creating search function...")
        create_search_function()
        
        # THEN create vector indexes after sentence_embeddings table exists
        print("Creating vector indexes...")
        create_vector_indexes()
    else:
        print("Skipping table creation (using existing tables)...")
    
    if not skip_loading:
        # Load data from Excel files
        print("Loading data from Excel files...")
        loaded_data = load_excel_data(data_folder)
        
        # Insert data with embeddings
        print("Inserting data with embeddings...")
        insert_data_with_embeddings(loaded_data)
        
        print("Database setup complete!")
    else:
        print("Skipping data loading (using existing data)...")
    
    if interactive:
        interactive_mode()

# Interactive mode for the student advisor bot
def interactive_mode():
    print("Welcome to the Student Advisor Bot!")
    print("Ask any question about courses, programs, or student information.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Search database
        try:
            results = search_database(user_input)
            
            if not results:
                print("No relevant information found in the database.")
            else:
                print(f"Found {len(results)} relevant pieces of information.")
                
                # Process with OpenAI
                response = process_query_with_openai(user_input, results)
                print("\nAdvisor Response:")
                print(response)
        except Exception as e:
            print(f"Error: {e}")
            print("Sorry, I encountered a problem. Please try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Student Advisor Database System')
    parser.add_argument('--skip-loading', action='store_true', help='Skip loading and embedding data (use existing data)')
    parser.add_argument('--skip-tables', action='store_true', help='Skip creating/recreating tables (use existing schema)')
    parser.add_argument('--interactive', action='store_true', help='Start in interactive mode immediately')
    parser.add_argument('--query-only', action='store_true', help='Run in query-only mode (equivalent to --skip-loading --skip-tables --interactive)')
    parser.add_argument('--similarity', type=float, default=0.3, help='Similarity threshold for vector search (0.0-1.0, default: 0.3)')
    parser.add_argument('--max-results', type=int, default=15, help='Maximum number of search results to return (default: 15)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  STUDENT ADVISOR SYSTEM")
    print("="*60)
    
    if args.query_only:
        print("\nRunning in query-only mode (no database modifications)")
        print("To modify the database or reload data, run without --query-only\n")
        print("For a better query experience with improved interface, try running query.py instead.\n")
    else:
        mode_description = []
        if args.skip_loading:
            mode_description.append("skip data loading")
        if args.skip_tables:
            mode_description.append("use existing tables")
        if args.interactive:
            mode_description.append("start in interactive mode")
            
        if mode_description:
            print(f"\nRunning with options: {', '.join(mode_description)}")
            
        if args.similarity != 0.3:
            print(f"Using custom similarity threshold: {args.similarity}")
        if args.max_results != 15:
            print(f"Using custom max results: {args.max_results}")
    
    try:
        if args.query_only:
            main(skip_loading=True, skip_tables=True, interactive=True)
        else:
            main(skip_loading=args.skip_loading, skip_tables=args.skip_tables, interactive=args.interactive)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFor help, run with --help flag: python pg_student_adv/pg2.py --help")