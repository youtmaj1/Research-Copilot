#!/bin/bash
# Research Copilot - Daily Morning Update Script
# Usage: ./daily_research.sh

echo "🔬 Research Copilot - Morning Update"
echo "===================================="

cd /Users/damian/Documents/projects/Research-Copilot

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🚀 Starting Ollama..."
    ollama serve &
    sleep 3
fi

echo "📚 Collecting latest papers..."

# 1. Update from ArXiv categories (last 24 hours)
echo "  • AI papers (cs.AI)..."
python -m collector.cli update cs.AI --days-back 1

echo "  • Machine Learning papers (cs.LG)..."
python -m collector.cli update cs.LG --days-back 1

echo "  • Computational Linguistics (cs.CL)..."
python -m collector.cli update cs.CL --days-back 1

# 2. Search for hot topics
echo "📄 Searching hot topics..."
python -c "
from collector import PaperCollector
collector = PaperCollector()

hot_topics = [
    'large language models 2024',
    'multimodal AI vision language',
    'efficient neural networks',
    'transformer optimization',
    'AI reasoning capabilities'
]

total_new = 0
for topic in hot_topics:
    try:
        results = collector.search(topic, max_results=3)
        new_papers = results['papers_added']
        total_new += new_papers
        if new_papers > 0:
            print(f'  • {topic}: {new_papers} new papers')
    except Exception as e:
        print(f'  • {topic}: Error - {e}')

print(f'\n📊 Total new papers from hot topics: {total_new}')
"

# 3. Show current statistics
echo ""
echo "📈 Current Collection Stats:"
python -m collector.cli stats

# 4. Show latest papers
echo ""
echo "📖 Latest Papers in Database:"
python -c "
import sqlite3
from datetime import datetime, timedelta

try:
    conn = sqlite3.connect('papers.db')
    cursor = conn.cursor()
    
    # Get papers from last 24 hours
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    cursor.execute('SELECT title, authors FROM papers WHERE ingested_at > ? ORDER BY ingested_at DESC LIMIT 5', (yesterday,))
    recent_papers = cursor.fetchall()
    
    if recent_papers:
        for i, (title, authors) in enumerate(recent_papers, 1):
            print(f'{i}. {title}')
            print(f'   Authors: {authors}')
            print()
    else:
        print('No new papers in the last 24 hours.')
    
    conn.close()
except Exception as e:
    print(f'Error accessing database: {e}')
"

echo "✅ Morning research update complete!"
echo ""
echo "🔍 Ready to start research session:"
echo "   • Web interface: streamlit run app.py"
echo "   • Ask questions about your papers using phi4-mini:3.8b"
echo ""