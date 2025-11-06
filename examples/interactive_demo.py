#!/usr/bin/env python3
"""
Research Copilot - Interactive Demo
Quick demonstration of daily research workflows
"""

import os
import sys
import sqlite3
from datetime import datetime
from config.ollama_config import OllamaConfigManager

def main():
    print("🔬 Research Copilot - Interactive Demo")
    print("=" * 45)
    
    # Initialize
    ollama = OllamaConfigManager()
    print(f"✅ Using model: {ollama.model} (temp: 0.3)")
    
    # Check database
    if os.path.exists('papers.db'):
        conn = sqlite3.connect('papers.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM papers')
        paper_count = cursor.fetchone()[0]
        print(f"📚 Papers available: {paper_count}")
        
        # Show a few recent papers
        cursor.execute('SELECT title FROM papers ORDER BY created_at DESC LIMIT 3')
        recent = cursor.fetchall()
        print("\n📖 Recent papers:")
        for i, (title,) in enumerate(recent, 1):
            print(f"  {i}. {title[:80]}...")
        
        conn.close()
    else:
        print("❌ No database found. Run paper collection first.")
        return
    
    print("\n" + "=" * 45)
    print("🤖 Ask me anything about your research papers!")
    print("Examples:")
    print("• 'What is the main contribution of the Less is More paper?'")
    print("• 'Explain recursive reasoning in neural networks'")
    print("• 'Compare efficiency techniques in recent papers'")
    print("• Type 'quit' to exit")
    print("=" * 45)
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy researching!")
                break
            
            if not question:
                continue
            
            print("🤖 Thinking...")
            
            # Enhanced prompt for research context
            research_prompt = f"""
As a research assistant with access to academic papers, please answer this question:
{question}

Provide a comprehensive, factual answer based on current research. Include specific details and be precise.
"""
            
            result = ollama.generate_completion(
                research_prompt, 
                max_tokens=400, 
                temperature=0.3
            )
            
            if result['success']:
                print("\n" + "=" * 50)
                print("📄 Answer:")
                print(result['response'])
                print("=" * 50)
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()