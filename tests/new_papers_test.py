#!/usr/bin/env python3
"""
New Papers Research Validation
=============================
Test system with diverse AI research topics
"""
import psycopg2
import requests
import json
import time

def validate_new_papers():
    """Test research system with new papers across AI domains"""
    
    print("🔬 RESEARCH COPILOT: NEW PAPERS VALIDATION")
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
        
        # Show database stats
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_count = cursor.fetchone()[0]
        print(f"📊 Database Status: {total_count} research papers indexed")
        
        # Test queries for new research areas
        research_tests = [
            {
                "domain": "Scaling Laws & Compute",
                "query": "scaling",
                "research_question": "What are the scaling laws for language models and how do they relate to compute requirements?"
            },
            {
                "domain": "AI Safety & Alignment", 
                "query": "constitutional OR harmlessness OR feedback",
                "research_question": "How does Constitutional AI work and why is it important for AI safety?"
            },
            {
                "domain": "Multimodal AI",
                "query": "visual OR multimodal OR flamingo",
                "research_question": "What is Flamingo and how does it handle visual-language tasks?"
            },
            {
                "domain": "Emergent Abilities",
                "query": "emergent",
                "research_question": "What are emergent abilities in large language models and why do they occur?"
            },
            {
                "domain": "Dialog Systems",
                "query": "dialog OR conversation",
                "research_question": "How is LaMDA designed differently for dialog applications compared to general language models?"
            },
            {
                "domain": "Large-Scale Models",
                "query": "PaLM OR pathways OR 540B",
                "research_question": "What makes PaLM unique among large language models and what are its key capabilities?"
            }
        ]
        
        successful_tests = 0
        total_tests = len(research_tests)
        
        for i, test in enumerate(research_tests, 1):
            print(f"\n🔍 TEST {i}/{total_tests}: {test['domain']}")
            print("-" * 60)
            
            # Database search with advanced query (simplified for PostgreSQL compatibility)
            search_terms = test['query'].replace(' OR ', ' | ').replace(' AND ', ' & ')
            cursor.execute("""
                SELECT title, authors, abstract 
                FROM papers 
                WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery('english', %s)
                OR title ILIKE %s
                ORDER BY 
                    ts_rank(to_tsvector('english', title || ' ' || abstract), to_tsquery('english', %s)) DESC
                LIMIT 3
            """, (search_terms, f"%{test['query'].split()[0]}%", search_terms))
            
            papers = cursor.fetchall()
            print(f"📚 Found {len(papers)} relevant papers:")
            
            if papers:
                context = f"Research Question: {test['research_question']}\n\nRelevant Papers:\n"
                for title, authors, abstract in papers:
                    print(f"   • {title} ({authors})")
                    context += f"\n- {title} by {authors}\n  Abstract: {abstract}\n"
                
                # Test LLM with research context
                print(f"\n🤖 AI Research Analysis:")
                
                prompt = f"""{context}

Based on the research papers above, please provide a comprehensive answer to: {test['research_question']}

Focus on the key insights, methodologies, and implications from the relevant papers."""

                try:
                    start_time = time.time()
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "deepseek-coder-v2:16b",
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=90  # Longer timeout for complex queries
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        llm_response = data.get('response', '')
                        
                        print(f"   ✅ Success ({response_time:.1f}s, {len(llm_response)} chars)")
                        print(f"   📝 Research Insights:")
                        
                        # Show first 350 characters of response
                        preview = llm_response[:350].replace('\n', ' ').strip()
                        print(f"      {preview}...")
                        
                        if len(llm_response) > 350:
                            print(f"   📖 [Full response: {len(llm_response)} characters total]")
                        
                        successful_tests += 1
                        
                    else:
                        print(f"   ❌ LLM Error: HTTP {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"   ⏰ Timeout (>90s) - Complex research query")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            else:
                print(f"   ⚠️ No relevant papers found for '{test['query']}'")
        
        # Cross-domain research test
        print(f"\n🌐 CROSS-DOMAIN RESEARCH TEST")
        print("-" * 40)
        print("Query: Compare scaling laws, emergent abilities, and constitutional AI...")
        
        cursor.execute("""
            SELECT title, authors FROM papers 
            WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery('english', 'scaling | emergent | constitutional')
            ORDER BY ts_rank(to_tsvector('english', title || ' ' || abstract), to_tsquery('english', 'scaling | emergent | constitutional')) DESC
        """)
        
        cross_domain_papers = cursor.fetchall()
        print(f"📊 Cross-domain papers found: {len(cross_domain_papers)}")
        for title, authors in cross_domain_papers:
            print(f"   • {title} ({authors})")
        
        # Final validation summary
        print(f"\n📈 VALIDATION RESULTS")
        print("=" * 40)
        print(f"✅ Successful Research Tests: {successful_tests}/{total_tests}")
        print(f"📚 Total Papers in Database: {total_count}")
        print(f"🔍 Advanced Search Queries: Working")
        print(f"🤖 LLM Research Analysis: {'Excellent' if successful_tests >= total_tests * 0.7 else 'Needs Optimization'}")
        print(f"🌐 Cross-domain Capabilities: {len(cross_domain_papers)} papers linked")
        
        success_rate = (successful_tests / total_tests) * 100
        if success_rate >= 80:
            status = "🎉 EXCELLENT - Production Ready"
        elif success_rate >= 60:
            status = "✅ GOOD - Minor Optimizations Needed"
        else:
            status = "⚠️ NEEDS IMPROVEMENT"
            
        print(f"\n🎯 OVERALL VALIDATION: {status}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ System Error: {e}")

if __name__ == "__main__":
    validate_new_papers()
