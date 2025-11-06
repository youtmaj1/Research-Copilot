"""
Advanced Paper Testing with Recursive Learning and Self-Improving Systems
========================================================================

This script will download recent papers on:
1. Gödel Machines and recursive self-improvement
2. Meta-learning and self-modifying systems
3. Recursive neural architectures
4. Self-improving AI systems

Then test the Research Copilot system with challenging questions.
"""

import os
import requests
import time
import json
from typing import List, Dict, Any
import sqlite3
from datetime import datetime

class PaperDownloader:
    """Download papers from arXiv on specific topics"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.papers = []
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for papers on arXiv"""
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
                
                if title_elem is not None and summary_elem is not None:
                    # Extract authors
                    authors = []
                    for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                        name_elem = author.find('.//{http://www.w3.org/2005/Atom}name')
                        if name_elem is not None:
                            authors.append(name_elem.text)
                    
                    paper = {
                        'id': id_elem.text.split('/')[-1] if id_elem is not None else f"paper_{len(papers)}",
                        'title': title_elem.text.strip(),
                        'abstract': summary_elem.text.strip(),
                        'authors': ', '.join(authors),
                        'published_date': published_elem.text if published_elem is not None else datetime.now().isoformat(),
                        'venue': 'arXiv',
                        'url': id_elem.text if id_elem is not None else '',
                        'query_topic': query
                    }
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []
    
    def download_recursive_learning_papers(self) -> List[Dict]:
        """Download papers on recursive learning and self-improving systems"""
        print("🔍 Downloading papers on recursive learning and self-improving systems...")
        
        queries = [
            "recursive self-improvement AI",
            "Gödel machine self-modifying",
            "meta-learning recursive neural",
            "self-improving artificial intelligence",
            "recursive optimization learning",
            "self-modifying neural networks",
            "bootstrap learning systems",
            "recursive neural architecture search"
        ]
        
        all_papers = []
        for query in queries:
            print(f"  Searching: {query}")
            papers = self.search_papers(query, max_results=5)
            all_papers.extend(papers)
            time.sleep(1)  # Be respectful to arXiv
        
        # Remove duplicates based on title similarity
        unique_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            title_key = paper['title'].lower().replace(' ', '')
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        print(f"✅ Downloaded {len(unique_papers)} unique papers")
        return unique_papers

def add_papers_to_database(papers: List[Dict], db_path: str = "papers.db"):
    """Add papers to the Research Copilot database"""
    print(f"📚 Adding {len(papers)} papers to database...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create papers table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                published_date TEXT,
                venue TEXT,
                url TEXT,
                pdf_path TEXT,
                keywords TEXT
            )
        """)
        
        # Insert papers
        for paper in papers:
            # Create a unique hash for deduplication
            import hashlib
            paper_hash = hashlib.md5(paper['title'].encode()).hexdigest()
            
            cursor.execute("""
                INSERT OR REPLACE INTO papers 
                (id, title, abstract, authors, published_date, venue, url, source, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper['id'],
                paper['title'],
                paper['abstract'],
                paper['authors'],
                paper['published_date'],
                paper['venue'],
                paper['url'],
                'arxiv',
                paper_hash
            ))
        
        conn.commit()
        print(f"✅ Added {len(papers)} papers to database")

def test_challenging_questions():
    """Test the system with challenging questions about recursive learning"""
    print("\n🧠 Testing Research Copilot with Challenging Questions")
    print("=" * 60)
    
    # Import the Q&A pipeline
    from qa.advanced_qa_pipeline import EnterpriseQAPipeline
    
    # Initialize pipeline
    qa_pipeline = EnterpriseQAPipeline()
    
    # Challenging questions about recursive learning and self-improvement
    challenging_questions = [
        {
            "question": "What are the theoretical foundations of Gödel machines and how do they enable provably optimal self-improvement?",
            "topic": "Gödel Machines Theory",
            "difficulty": "Expert"
        },
        {
            "question": "How do recursive self-improving systems avoid the optimization daemon problem and maintain goal stability during self-modification?",
            "topic": "AI Safety in Self-Improvement",
            "difficulty": "Expert"
        },
        {
            "question": "What are the key differences between meta-learning, recursive neural architecture search, and true recursive self-improvement in AI systems?",
            "topic": "Recursive Learning Taxonomy",
            "difficulty": "Advanced"
        },
        {
            "question": "How do bootstrap learning systems overcome the cold start problem and what mathematical guarantees exist for convergence?",
            "topic": "Bootstrap Learning Theory",
            "difficulty": "Expert"
        },
        {
            "question": "What are the computational complexity bounds for recursive optimization in self-modifying neural networks?",
            "topic": "Computational Complexity",
            "difficulty": "Expert"
        },
        {
            "question": "How do current implementations of self-improving AI systems handle the exploration-exploitation trade-off during recursive optimization?",
            "topic": "Exploration-Exploitation",
            "difficulty": "Advanced"
        },
        {
            "question": "What role does formal verification play in ensuring the safety of self-modifying AI systems, and what are the current limitations?",
            "topic": "Formal Verification & Safety",
            "difficulty": "Expert"
        },
        {
            "question": "How do recursive learning systems maintain interpretability and explainability as they self-modify their architecture and parameters?",
            "topic": "Interpretability in Self-Improvement",
            "difficulty": "Advanced"
        }
    ]
    
    results = []
    
    for i, qa_item in enumerate(challenging_questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(challenging_questions)} - {qa_item['topic']} ({qa_item['difficulty']})")
        print(f"{'='*80}")
        print(f"❓ {qa_item['question']}")
        print(f"\n🤖 Ollama DeepSeek-Coder-V2:16B Response:")
        print("-" * 60)
        
        start_time = time.time()
        
        # Get answer from the system
        response = qa_pipeline.answer_question(qa_item['question'])
        
        response_time = time.time() - start_time
        
        print(f"📝 Answer: {response.answer}")
        print(f"\n📊 Metrics:")
        print(f"  • Confidence: {response.confidence:.3f}")
        print(f"  • Response Time: {response_time:.2f}s")
        print(f"  • Retrieved Chunks: {response.retrieved_chunks}")
        print(f"  • Citations: {len(response.citations)}")
        print(f"  • Method Used: {response.method_used}")
        
        if response.citations:
            print(f"\n📚 Key Citations:")
            for j, citation in enumerate(response.citations[:3], 1):
                print(f"  {j}. {citation.paper_title} (Relevance: {citation.relevance_score:.3f})")
        
        results.append({
            'question': qa_item['question'],
            'topic': qa_item['topic'],
            'difficulty': qa_item['difficulty'],
            'answer': response.answer,
            'confidence': response.confidence,
            'response_time': response_time,
            'citations_count': len(response.citations),
            'retrieved_chunks': response.retrieved_chunks
        })
        
        # Brief pause between questions
        time.sleep(2)
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("📈 TESTING SUMMARY")
    print(f"{'='*80}")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_response_time = sum(r['response_time'] for r in results) / len(results)
    total_citations = sum(r['citations_count'] for r in results)
    
    print(f"Questions Tested: {len(results)}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print(f"Total Citations Generated: {total_citations}")
    
    # Confidence by difficulty
    expert_questions = [r for r in results if r['difficulty'] == 'Expert']
    advanced_questions = [r for r in results if r['difficulty'] == 'Advanced']
    
    if expert_questions:
        expert_confidence = sum(r['confidence'] for r in expert_questions) / len(expert_questions)
        print(f"Expert Question Confidence: {expert_confidence:.3f}")
    
    if advanced_questions:
        advanced_confidence = sum(r['confidence'] for r in advanced_questions) / len(advanced_questions)
        print(f"Advanced Question Confidence: {advanced_confidence:.3f}")
    
    return results

def main():
    """Main testing workflow"""
    print("🧠 Advanced Research Copilot Testing: Recursive Learning & Self-Improving Systems")
    print("=" * 90)
    
    # Step 1: Download papers
    downloader = PaperDownloader()
    papers = downloader.download_recursive_learning_papers()
    
    if not papers:
        print("❌ No papers downloaded. Using existing database.")
    else:
        # Step 2: Add papers to database
        add_papers_to_database(papers)
    
    # Step 3: Test with challenging questions
    results = test_challenging_questions()
    
    # Step 4: Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("-" * 40)
    
    high_confidence_count = sum(1 for r in results if r['confidence'] > 0.7)
    medium_confidence_count = sum(1 for r in results if 0.4 <= r['confidence'] <= 0.7)
    low_confidence_count = sum(1 for r in results if r['confidence'] < 0.4)
    
    print(f"High Confidence (>0.7): {high_confidence_count}")
    print(f"Medium Confidence (0.4-0.7): {medium_confidence_count}")
    print(f"Low Confidence (<0.4): {low_confidence_count}")
    
    if high_confidence_count >= len(results) * 0.6:
        print("🏆 EXCELLENT: System demonstrates strong understanding of complex topics")
    elif medium_confidence_count + high_confidence_count >= len(results) * 0.7:
        print("✅ GOOD: System shows solid performance on challenging questions")
    else:
        print("📚 LEARNING: System could benefit from more domain-specific papers")
    
    print(f"\n🦙 Ollama DeepSeek-Coder-V2:16B Performance on Recursive Learning: TESTED")
    print(f"📊 Advanced Cross-Referencing and RAG Accuracy: EVALUATED")

if __name__ == "__main__":
    main()
