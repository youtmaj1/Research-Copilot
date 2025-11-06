# 🔬 Research Copilot - Daily Usage Guide

Your complete guide to using Research Copilot for daily research paper discovery, collection, and analysis.

## 🚀 Quick Start (5 minutes)

### Option 1: Web Interface (Recommended)
```bash
cd /Users/damian/Documents/projects/Research-Copilot
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### Option 2: Command Line Interface
```bash
cd /Users/damian/Documents/projects/Research-Copilot
python -m collector.cli --help
```

---

## 📋 Daily Workflow

### **Step 1: Collect New Papers (Morning Routine)**

#### A. Search for Specific Topics
```bash
# Search for papers on a specific topic
python -c "
from collector import PaperCollector
collector = PaperCollector()
results = collector.search('transformers attention mechanism', max_results=10)
print(f'✅ Collected {results[\"papers_added\"]} new papers')
"
```

#### B. Update by ArXiv Categories
```bash
# Get latest papers from AI/ML categories
python -m collector.cli update cs.AI --days-back 1
python -m collector.cli update cs.LG --days-back 1
python -m collector.cli update cs.CL --days-back 1
```

#### C. Targeted Search Examples
```bash
# Recent breakthrough papers
python -c "
from collector import PaperCollector
import datetime
collector = PaperCollector()

# Search topics you're interested in
topics = [
    'large language models reasoning',
    'multimodal AI vision language',
    'efficient neural networks',
    'AI safety alignment',
    'recursive reasoning tiny networks'
]

for topic in topics:
    results = collector.search(topic, max_results=5)
    print(f'📄 {topic}: {results[\"papers_added\"]} papers')
"
```

### **Step 2: Review Your Collection**

#### Check What You Have
```bash
python -c "
import sqlite3
conn = sqlite3.connect('papers.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM papers')
total = cursor.fetchone()[0]
print(f'📚 Total papers: {total}')

# Recent papers
cursor.execute('SELECT title, authors FROM papers ORDER BY ingested_at DESC LIMIT 5')
recent = cursor.fetchall()
print('\n📖 Latest papers:')
for title, authors in recent:
    print(f'• {title} by {authors}')
conn.close()
"
```

#### Get Statistics
```bash
python -m collector.cli stats
```

### **Step 3: Interactive Q&A Session**

#### A. Start the Web Interface
```bash
streamlit run app.py
```

**In the web interface:**
1. 🔧 **Configure** (sidebar):
   - LLM Model: `phi4-mini:3.8b` (already set)
   - Temperature: `0.3` for factual answers
   - Max chunks: `5-10` for comprehensive answers

2. 🔍 **Ask Questions**:
   - Type your research question
   - Get AI-powered answers with citations
   - Follow up with more specific questions

#### B. Command Line Q&A
```bash
python -c "
from config.ollama_config import OllamaConfigManager

ollama = OllamaConfigManager()
question = input('🤔 What would you like to know about your papers? ')
response = ollama.generate_completion(f'''
Based on research papers in machine learning and AI, answer this question:
{question}

Provide a comprehensive, factual answer.
''', max_tokens=500, temperature=0.3)

print('🤖 Answer:')
print('=' * 50)
print(response['response'])
"
```

---

## 💡 **Sample Daily Workflows**

### **Morning Research Routine (10 minutes)**

```bash
#!/bin/bash
# Save as daily_research.sh

echo "🌅 Morning Research Update"
cd /Users/damian/Documents/projects/Research-Copilot

# 1. Get latest papers
python -m collector.cli update cs.AI --days-back 1
python -m collector.cli update cs.LG --days-back 1

# 2. Search hot topics
python -c "
from collector import PaperCollector
collector = PaperCollector()
hot_topics = ['GPT-4', 'multimodal AI', 'efficient transformers']
for topic in hot_topics:
    results = collector.search(topic, max_results=3)
    print(f'📄 {topic}: {results[\"papers_added\"]} new papers')
"

# 3. Show stats
python -m collector.cli stats

echo "✅ Research update complete!"
```

Make it executable:
```bash
chmod +x daily_research.sh
./daily_research.sh
```

### **Deep Dive Session (30-60 minutes)**

1. **Start Web Interface**:
   ```bash
   streamlit run app.py
   ```

2. **Research Questions to Ask**:
   - "What are the latest breakthroughs in transformer architectures?"
   - "Compare different approaches to model efficiency in recent papers"
   - "What are the key findings about recursive reasoning in small models?"
   - "Summarize the main contributions of papers published this week"

3. **Follow-up Questions**:
   - "Can you elaborate on the methodology used in [specific paper]?"
   - "What are the limitations mentioned in recent efficiency papers?"
   - "How do these findings compare to earlier work?"

---

## 🎯 **Advanced Usage Patterns**

### **Topic-Focused Research**

```bash
# Create a focused collection on a specific topic
python -c "
from collector import PaperCollector
collector = PaperCollector()

# Your research focus
focus_topic = 'neural architecture search'

# Comprehensive search
results = collector.search(f'{focus_topic} 2024', max_results=20)
results2 = collector.search(f'{focus_topic} efficiency', max_results=15)
results3 = collector.search(f'{focus_topic} transformers', max_results=10)

total = results['papers_added'] + results2['papers_added'] + results3['papers_added']
print(f'📚 Collected {total} papers on {focus_topic}')
"
```

### **Weekly Review Session**

```bash
python -c "
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('papers.db')
cursor = conn.cursor()

# Papers from last week
week_ago = (datetime.now() - timedelta(days=7)).isoformat()
cursor.execute('SELECT title, authors FROM papers WHERE ingested_at > ? ORDER BY ingested_at DESC', (week_ago,))
papers = cursor.fetchall()

print(f'📊 Papers collected in the last week: {len(papers)}')
print('\n📖 Recent additions:')
for title, authors in papers[:10]:
    print(f'• {title}')

conn.close()
"
```

---

## 🔧 **Customization & Tips**

### **Adjust LLM Behavior**

For different types of questions, you can adjust the temperature:

```python
# For creative/brainstorming (higher temperature)
response = ollama.generate_completion(question, temperature=0.7)

# For factual/precise answers (lower temperature) - Default
response = ollama.generate_completion(question, temperature=0.3)

# For very precise/deterministic (lowest temperature)
response = ollama.generate_completion(question, temperature=0.1)
```

### **Search Strategy Tips**

1. **Use specific keywords**: "transformer attention mechanism" vs "AI"
2. **Include year**: "GPT 2024" for recent papers
3. **Combine terms**: "efficiency AND transformer AND optimization"
4. **Author names**: "Vaswani attention mechanism" for specific researchers

### **Question Strategies**

**Good Questions:**
- "What are the key innovations in the Less is More paper?"
- "Compare recent approaches to model compression"
- "What evaluation metrics are used in efficiency papers?"

**Great Follow-ups:**
- "Can you explain the methodology in more detail?"
- "What are the limitations of this approach?"
- "How does this compare to previous work?"

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

1. **No papers found**: Try broader search terms
2. **LLM not responding**: Check `ollama list` and restart if needed
3. **Slow responses**: Reduce max_chunks in web interface
4. **Database errors**: Check if `papers.db` exists and has proper permissions

### **Restart Everything**
```bash
# If something goes wrong
pkill -f streamlit
pkill -f ollama
ollama serve  # Start ollama
streamlit run app.py  # Start web interface
```

---

## 📅 **Suggested Daily Schedule**

### **Morning (5-10 minutes)**
- Run daily collection script
- Quick browse of new papers
- Note interesting titles for deep dive

### **Mid-day (15-30 minutes)**
- Open web interface
- Ask 2-3 research questions
- Follow up on interesting findings

### **Evening (Optional - 30-60 minutes)**
- Deep dive session on specific papers
- Comprehensive Q&A on complex topics
- Plan tomorrow's research focus

---

## 🎉 **You're Ready!**

Your Research Copilot with **phi4-mini:3.8b** at **temperature 0.3** is optimized for:
- ✅ **Factual accuracy**
- ✅ **Research-focused responses** 
- ✅ **Efficient daily workflows**
- ✅ **Interactive exploration**

**Start with**: `streamlit run app.py` and ask about the "Less is More: Recursive Reasoning with Tiny Networks" paper you just collected!