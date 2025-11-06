#!/usr/bin/env python3
"""
Direct Research Query Test
=========================
Test research queries directly with database integration
"""
import psycopg2
import requests
import json

def direct_research_test():
    """Test research functionality with database search + LLM"""
    
    print("🔬 DIRECT RESEARCH QUERY TEST")
    print("=" * 40)
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="research_copilot",
            user="research_user",
            password="research_password"
        )
        print("✅ Database connected")
        
        # Test research queries
        test_queries = [
            "chain-of-thought",
            "GPT-4",
            "retrieval augmented",
            "language models"
        ]
        
        for query in test_queries:
            print(f"\n📝 Searching for: '{query}'")
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT title, authors, abstract 
                FROM papers 
                WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery('english', %s)
                LIMIT 3
            """, (query.replace(' ', ' & '),))
            
            results = cursor.fetchall()
            print(f"   📚 Found {len(results)} relevant papers")
            
            for i, (title, authors, abstract) in enumerate(results, 1):
                print(f"   {i}. {title} - {authors}")
                print(f"      {abstract[:100]}...")
            
            cursor.close()
        
        conn.close()
        
        # Test Ollama directly
        print(f"\n🤖 Testing Ollama Integration")
        
        research_prompt = """Based on the research papers in the database about language models, 
        explain what chain-of-thought prompting is and how it relates to reasoning in AI."""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-coder-v2:16b",
                "prompt": research_prompt,
                "stream": False
            },
            timeout=45
        )
        
        if response.status_code == 200:
            data = response.json()
            llm_response = data.get('response', '')
            print(f"✅ LLM Response Length: {len(llm_response)} chars")
            print(f"🔍 Response Preview:")
            print(f"   {llm_response[:300]}...")
        else:
            print(f"❌ LLM Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    direct_research_test()
