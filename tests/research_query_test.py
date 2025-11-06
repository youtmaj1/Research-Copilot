#!/usr/bin/env python3
"""
Research Paper Query System
==========================
Test the system with research-specific queries
"""
import requests
import json
import time

def test_research_queries():
    """Test system with research paper queries"""
    
    print("🔬 RESEARCH PAPER QUERY TESTING")
    print("=" * 50)
    
    # Test queries related to our paper database
    test_queries = [
        {
            "query": "What is chain-of-thought prompting?",
            "expected_topics": ["reasoning", "prompting", "language models"]
        },
        {
            "query": "Compare GPT-4 with LLaMA models",
            "expected_topics": ["GPT-4", "LLaMA", "comparison"]
        },
        {
            "query": "What is retrieval-augmented generation?",
            "expected_topics": ["RAG", "retrieval", "generation"]
        },
        {
            "query": "How does InstructGPT work?",
            "expected_topics": ["InstructGPT", "human feedback", "alignment"]
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📝 TEST QUERY {i}: {test['query']}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # First attempt (cache miss expected)
            response = requests.post(
                "http://localhost:8000/api/v1/query",
                json={"query": test['query']},
                timeout=45
            )
            
            query_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ SUCCESS (HTTP {response.status_code})")
                print(f"⏱️  Response Time: {data.get('response_time', query_time):.2f}s")
                print(f"💾 Cached: {data.get('cached', False)}")
                print(f"🤖 Model: {data.get('model', 'N/A')}")
                
                response_text = data.get('response', '')
                print(f"📄 Response Length: {len(response_text)} characters")
                print(f"🔍 Response Preview:")
                print(f"   {response_text[:300]}...")
                
                # Test caching by making the same query again
                print(f"\n🔄 Testing Cache Performance...")
                cache_start = time.time()
                cache_response = requests.post(
                    "http://localhost:8000/api/v1/query",
                    json={"query": test['query']},
                    timeout=10
                )
                cache_time = time.time() - cache_start
                
                if cache_response.status_code == 200:
                    cache_data = cache_response.json()
                    cached = cache_data.get('cached', False)
                    print(f"   Cache Hit: {cached}")
                    print(f"   Cache Time: {cache_time:.3f}s")
                    if not cached:
                        print(f"   ⚠️ Expected cache hit but got cache miss")
                
                results.append({
                    'query': test['query'],
                    'success': True,
                    'response_time': data.get('response_time', query_time),
                    'cached': data.get('cached', False),
                    'response_length': len(response_text)
                })
                
            else:
                print(f"❌ FAILED (HTTP {response.status_code})")
                print(f"   Error: {response.text}")
                results.append({
                    'query': test['query'],
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                })
                
        except requests.exceptions.Timeout:
            print(f"⏰ TIMEOUT after 45 seconds")
            results.append({
                'query': test['query'],
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'query': test['query'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 RESEARCH QUERY TEST SUMMARY")
    print("=" * 50)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful Queries: {len(successful)}/{len(results)}")
    print(f"❌ Failed Queries: {len(failed)}/{len(results)}")
    
    if successful:
        avg_time = sum(r['response_time'] for r in successful) / len(successful)
        print(f"⏱️  Average Response Time: {avg_time:.2f}s")
        print(f"📄 Average Response Length: {sum(r['response_length'] for r in successful) / len(successful):.0f} chars")
    
    if failed:
        print(f"\n❌ Failed Query Details:")
        for fail in failed:
            print(f"   • {fail['query']}: {fail['error']}")
    
    return results

if __name__ == "__main__":
    test_research_queries()
