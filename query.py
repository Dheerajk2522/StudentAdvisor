import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
import time
from openai import OpenAI
import sys

# Import necessary functions from pg2.py
from SC.pg2 import (
    DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT,
    engine, client, generate_embedding, store_query_embedding,
    search_database, process_query_with_openai
)

def interactive_mode():
    """Interactive mode for the student advisor bot"""
    print("\n" + "="*60)
    print("  WELCOME TO THE STUDENT ADVISOR BOT")
    print("="*60)
    print("\nI can help with academic advising questions such as:")
    print("  • Course recommendations for specific students")
    print("  • Prerequisites and study plans")
    print("  • Course scheduling and availability")
    print("  • Program requirements and specializations")
    print("\nExamples:")
    print("  > Which courses should [student name] take next semester?")
    print("  > What are the prerequisites for [course code]?")
    print("  > How can a student specialize in [specialization area]?")
    print("\nType 'exit' to quit.")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\n🎓 Your question: ")
            
            if not user_input.strip():
                print("Please enter a question.")
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Student Advisor Bot. Goodbye!")
                break
            
            print("\n🔍 Searching database for relevant information...")
            
            # Search database
            try:
                start_time = time.time()
                results = search_database(user_input)
                search_time = time.time() - start_time
                
                if not results:
                    print("⚠️ No relevant information found in the database.")
                    print("Try asking a different question related to student courses, programs, or enrollment.")
                else:
                    source_counts = {}
                    for r in results:
                        source = r['source_table']
                        source_counts[source] = source_counts.get(source, 0) + 1
                    
                    sources_str = ", ".join([f"{count} {source}" for source, count in source_counts.items()])
                    print(f"✅ Found {len(results)} relevant pieces of information across {len(source_counts)} sources in {search_time:.2f} seconds.")
                    print(f"   Sources: {sources_str}")
                    
                    # Process with OpenAI
                    print("\n⚙️ Generating detailed response...")
                    start_time = time.time()
                    response = process_query_with_openai(user_input, results)
                    response_time = time.time() - start_time
                    
                    print(f"\n📝 ADVISOR RESPONSE (generated in {response_time:.2f} seconds):")
                    print("-"*60)
                    print(response)
                    print("-"*60)
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Sorry, I encountered a problem with the database search or AI processing.")
                print("Please try again with a different question or check your database connection.")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    print("Student Advisor Query Tool")
    print("=========================")
    print("This tool allows you to query the existing database without re-embedding data.")
    print("Note: Make sure you've run the full application (pg2.py) at least once to create embeddings.")
    
    try:
        interactive_mode()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please ensure the database has been set up correctly by running pg2.py first.")
        sys.exit(1) 