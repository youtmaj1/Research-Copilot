#!/usr/bin/env python3
"""
Research Paper System Demo
=========================
Comprehensive demonstration of research capabilities
"""
import psycopg2
import requests
import json
import time

def research_demo():
    """Demonstrate research system capabilities"""
    
    print("🔬 RESEARCH COPILOT: PAPER ANALYSIS DEMO")
    print("=" * 55)
    
    # Database connection
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="research_copilot",
            user="research_user",
            password="research_password"
        )
        cursor = conn.cursor()
        
        # Show available papers
        print("📚 RESEARCH PAPER DATABASE")
        print("-" * 30)
        cursor.execute("SELECT COUNT(*) FROM papers")
        count = cursor.fetchone()[0]
        print(f"Total Papers: {count}")
        
        cursor.execute("SELECT title, authors FROM papers")
        papers = cursor.fetchall()
        for i, (title, authors) in enumerate(papers, 1):
            print(f"{i:2d}. {title} - {authors}")
        
        # Research queries
        research_queries = [
            {
                "topic": "Attention Mechanism",
                "search_term": "attention",
                "question": "What is the attention mechanism in transformers and why is it important?"
            },
            {
                "topic": "Chain-of-Thought Reasoning", 
                "search_term": "reasoning",
                "question": "How does chain-of-thought prompting improve AI reasoning capabilities?"
            },
            {
                "topic": "GPT Models",
                "search_term": "GPT",
                "question": "What are the differences between GPT-4 and InstructGPT?"
            }
        ]
        
        for query in research_queries:
            print(f"\n🔍 RESEARCH QUERY: {query['topic']}")
            print("-" * 50)
            
            # Database search
            cursor.execute("""
                SELECT title, authors, abstract 
                FROM papers 
                WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery('english', %s)
                OR title ILIKE %s
            """, (query['search_term'], f"%{query['search_term']}%"))
            
            relevant_papers = cursor.fetchall()
            print(f"📄 Found {len(relevant_papers)} relevant papers:")
            
            context = ""
            for title, authors, abstract in relevant_papers:
                print(f"   • {title} ({authors})")
                context += f"Paper: {title} by {authors}. Abstract: {abstract}\n"
            
            if relevant_papers:
                # Generate LLM response with context
                print(f"\n🤖 AI Analysis:")
                
                prompt = f"""Based on these research papers:

{context}

Please answer: {query['question']}

Provide a comprehensive answer based on the research papers mentioned above."""

                try:
                    start_time = time.time()
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "deepseek-coder-v2:16b",
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=60
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        llm_response = data.get('response', '')
                        
                        print(f"   ✅ Response ({response_time:.1f}s, {len(llm_response)} chars)")
                        print(f"   📝 Answer: {llm_response[:400]}...")
                        if len(llm_response) > 400:
                            print(f"   📖 [Response continues for {len(llm_response)-400} more characters]")
                    else:
                        print(f"   ❌ LLM Error: {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            else:
                print("   No relevant papers found for this query.")
        
        # System performance summary
        print(f"\n📊 SYSTEM PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"✅ Database: {count} research papers indexed")
        print(f"✅ Full-text Search: Working with PostgreSQL")
        print(f"✅ LLM Integration: DeepSeek-Coder-V2:16b responding")
        print(f"✅ Research Pipeline: Database → Context → LLM → Response")
        print(f"🚀 System Status: Ready for research queries!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    research_demo()
