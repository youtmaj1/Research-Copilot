"""
Answer Formatter for RAG Pipeline

Formats RAG pipeline answers with proper JSON structure, citations,
and enhanced presentation for display in various interfaces.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FormattedCitation:
    """Structured citation information."""
    id: str
    title: str
    authors: str
    doi: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]
    year: Optional[int]

@dataclass
class FormattedAnswer:
    """Formatted answer with structured data."""
    answer: str
    citations: List[FormattedCitation]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: str

class AnswerFormatter:
    """
    Formats RAG pipeline answers for different output formats.
    
    Supports:
    - JSON formatting with structured citations
    - Markdown formatting for display
    - Plain text formatting
    - Citation link generation
    """
    
    def __init__(self, citation_style: str = "academic"):
        """
        Initialize answer formatter.
        
        Args:
            citation_style: Citation formatting style ('academic', 'brief', 'full')
        """
        self.citation_style = citation_style
        logger.info(f"Answer formatter initialized with style: {citation_style}")
    
    def _extract_paper_info(self, chunks, paper_id: str) -> Dict[str, Any]:
        """Extract paper information from chunks or metadata."""
        for chunk in chunks:
            if chunk.paper_id == paper_id:
                return {
                    'paper_id': paper_id,
                    'title': chunk.metadata.get('title', 'Unknown Title'),
                    'authors': chunk.metadata.get('authors', 'Unknown Authors'),
                    'doi': chunk.metadata.get('doi'),
                    'arxiv_id': chunk.metadata.get('arxiv_id'),
                    'year': chunk.metadata.get('year'),
                    'url': chunk.metadata.get('url')
                }
        
        # Fallback for unknown papers
        return {
            'paper_id': paper_id,
            'title': 'Unknown Title',
            'authors': 'Unknown Authors',
            'doi': None,
            'arxiv_id': None,
            'year': None,
            'url': None
        }
    
    def _parse_citation_id(self, citation_id: str) -> Dict[str, str]:
        """Parse citation ID to extract type and identifier."""
        citation_id = citation_id.strip()
        
        if citation_id.startswith('arxiv:'):
            return {'type': 'arxiv', 'id': citation_id[6:]}
        elif citation_id.startswith('doi:'):
            return {'type': 'doi', 'id': citation_id[4:]}
        elif citation_id.startswith('paper_'):
            return {'type': 'paper', 'id': citation_id[6:]}
        else:
            return {'type': 'unknown', 'id': citation_id}
    
    def _generate_citation_url(self, citation_info: Dict[str, str]) -> Optional[str]:
        """Generate URL for citation based on type."""
        if citation_info['type'] == 'arxiv':
            return f"https://arxiv.org/abs/{citation_info['id']}"
        elif citation_info['type'] == 'doi':
            return f"https://doi.org/{citation_info['id']}"
        else:
            return None
    
    def _format_citation_text(self, citation: FormattedCitation) -> str:
        """Format citation text based on style."""
        if self.citation_style == "brief":
            if citation.arxiv_id:
                return f"[{citation.arxiv_id}]"
            elif citation.doi:
                return f"[DOI: {citation.doi}]"
            else:
                return f"[{citation.id}]"
        
        elif self.citation_style == "full":
            parts = []
            if citation.authors != "Unknown Authors":
                parts.append(citation.authors)
            if citation.title != "Unknown Title":
                parts.append(f'"{citation.title}"')
            if citation.year:
                parts.append(f"({citation.year})")
            if citation.arxiv_id:
                parts.append(f"arXiv:{citation.arxiv_id}")
            elif citation.doi:
                parts.append(f"DOI:{citation.doi}")
            
            return ". ".join(parts) if parts else f"[{citation.id}]"
        
        else:  # academic style (default)
            if citation.authors != "Unknown Authors" and citation.year:
                return f"[{citation.authors.split(',')[0].strip()} et al., {citation.year}]"
            elif citation.arxiv_id:
                return f"[arXiv:{citation.arxiv_id}]"
            elif citation.doi:
                return f"[DOI:{citation.doi}]"
            else:
                return f"[{citation.id}]"
    
    def format_answer(
        self,
        answer: str,
        citations: List[str],
        chunks,
        confidence: float = 1.0
    ) -> str:
        """
        Format answer text with proper citation formatting.
        
        Args:
            answer: Raw answer text
            citations: List of citation IDs
            chunks: Retrieved chunks with metadata
            confidence: Answer confidence score
            
        Returns:
            Formatted answer string
        """
        # Create formatted citations
        formatted_citations = []
        
        for citation_id in citations:
            citation_info = self._parse_citation_id(citation_id)
            paper_info = self._extract_paper_info(chunks, citation_info['id'])
            
            formatted_citation = FormattedCitation(
                id=citation_id,
                title=paper_info['title'],
                authors=paper_info['authors'],
                doi=paper_info['doi'],
                arxiv_id=paper_info['arxiv_id'],
                url=self._generate_citation_url(citation_info),
                year=paper_info['year']
            )
            formatted_citations.append(formatted_citation)
        
        # Replace citation placeholders in answer
        formatted_answer = answer
        for citation in formatted_citations:
            citation_text = self._format_citation_text(citation)
            
            # Replace various citation formats
            patterns = [
                f"\\[{re.escape(citation.id)}\\]",
                f"\\({re.escape(citation.id)}\\)",
                re.escape(citation.id)
            ]
            
            for pattern in patterns:
                formatted_answer = re.sub(pattern, citation_text, formatted_answer, flags=re.IGNORECASE)
        
        return formatted_answer
    
    def format_as_json(
        self,
        answer: str,
        citations: List[str],
        chunks,
        confidence: float = 1.0,
        query: str = "",
        processing_time: float = 0.0
    ) -> str:
        """
        Format answer as JSON structure.
        
        Args:
            answer: Answer text
            citations: Citation IDs
            chunks: Retrieved chunks
            confidence: Confidence score
            query: Original query
            processing_time: Processing time in seconds
            
        Returns:
            JSON string
        """
        # Create formatted citations
        formatted_citations = []
        
        for citation_id in citations:
            citation_info = self._parse_citation_id(citation_id)
            paper_info = self._extract_paper_info(chunks, citation_info['id'])
            
            citation_dict = {
                "id": citation_id,
                "title": paper_info['title'],
                "authors": paper_info['authors'],
                "doi": paper_info['doi'],
                "arxiv_id": paper_info['arxiv_id'],
                "url": self._generate_citation_url(citation_info),
                "year": paper_info['year']
            }
            formatted_citations.append(citation_dict)
        
        # Create response structure
        response = {
            "answer": answer,
            "citations": formatted_citations,
            "metadata": {
                "query": query,
                "confidence": confidence,
                "processing_time": processing_time,
                "num_citations": len(citations),
                "num_chunks_used": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
    
    def format_as_markdown(
        self,
        answer: str,
        citations: List[str],
        chunks,
        confidence: float = 1.0,
        query: str = "",
        include_metadata: bool = True
    ) -> str:
        """
        Format answer as Markdown with citations.
        
        Args:
            answer: Answer text
            citations: Citation IDs
            chunks: Retrieved chunks
            confidence: Confidence score
            query: Original query
            include_metadata: Whether to include metadata section
            
        Returns:
            Markdown formatted string
        """
        markdown_parts = []
        
        # Add query if provided
        if query:
            markdown_parts.append(f"**Query:** {query}\n")
        
        # Add formatted answer
        formatted_answer = self.format_answer(answer, citations, chunks, confidence)
        markdown_parts.append(f"**Answer:**\n{formatted_answer}\n")
        
        # Add citations section
        if citations:
            markdown_parts.append("**References:**")
            
            for i, citation_id in enumerate(citations, 1):
                citation_info = self._parse_citation_id(citation_id)
                paper_info = self._extract_paper_info(chunks, citation_info['id'])
                url = self._generate_citation_url(citation_info)
                
                citation_line = f"{i}. {paper_info['title']}"
                if paper_info['authors'] != "Unknown Authors":
                    citation_line += f" - {paper_info['authors']}"
                if paper_info['year']:
                    citation_line += f" ({paper_info['year']})"
                if url:
                    citation_line += f" - [Link]({url})"
                
                markdown_parts.append(citation_line)
            
            markdown_parts.append("")  # Empty line
        
        # Add metadata if requested
        if include_metadata:
            markdown_parts.append("**Metadata:**")
            markdown_parts.append(f"- Confidence: {confidence:.2f}")
            markdown_parts.append(f"- Citations: {len(citations)}")
            markdown_parts.append(f"- Sources: {len(chunks)} chunks")
            markdown_parts.append("")
        
        return "\n".join(markdown_parts)
    
    def format_as_html(
        self,
        answer: str,
        citations: List[str],
        chunks,
        confidence: float = 1.0,
        query: str = ""
    ) -> str:
        """
        Format answer as HTML with clickable citations.
        
        Args:
            answer: Answer text
            citations: Citation IDs
            chunks: Retrieved chunks
            confidence: Confidence score
            query: Original query
            
        Returns:
            HTML formatted string
        """
        html_parts = []
        
        # Add query
        if query:
            html_parts.append(f'<h3>Query</h3><p><em>{query}</em></p>')
        
        # Format answer with HTML citations
        formatted_answer = answer
        citation_links = []
        
        for i, citation_id in enumerate(citations, 1):
            citation_info = self._parse_citation_id(citation_id)
            paper_info = self._extract_paper_info(chunks, citation_info['id'])
            url = self._generate_citation_url(citation_info)
            
            # Create citation link
            if url:
                citation_link = f'<a href="{url}" target="_blank" title="{paper_info["title"]}">[{i}]</a>'
            else:
                citation_link = f'<span title="{paper_info["title"]}">[{i}]</span>'
            
            citation_links.append(citation_link)
            
            # Replace citation in text
            patterns = [
                f"\\[{re.escape(citation_id)}\\]",
                f"\\({re.escape(citation_id)}\\)",
                re.escape(citation_id)
            ]
            
            for pattern in patterns:
                formatted_answer = re.sub(pattern, citation_link, formatted_answer, flags=re.IGNORECASE)
        
        html_parts.append(f'<h3>Answer</h3><p>{formatted_answer}</p>')
        
        # Add references section
        if citations:
            html_parts.append('<h3>References</h3><ol>')
            
            for i, citation_id in enumerate(citations, 1):
                citation_info = self._parse_citation_id(citation_id)
                paper_info = self._extract_paper_info(chunks, citation_info['id'])
                url = self._generate_citation_url(citation_info)
                
                citation_html = f'<li><strong>{paper_info["title"]}</strong>'
                if paper_info['authors'] != "Unknown Authors":
                    citation_html += f' - {paper_info["authors"]}'
                if paper_info['year']:
                    citation_html += f' ({paper_info["year"]})'
                if url:
                    citation_html += f' - <a href="{url}" target="_blank">Link</a>'
                citation_html += '</li>'
                
                html_parts.append(citation_html)
            
            html_parts.append('</ol>')
        
        # Add metadata
        html_parts.append(f'<div style="margin-top: 20px; font-size: 0.9em; color: #666;">')
        html_parts.append(f'Confidence: {confidence:.2f} | Citations: {len(citations)} | Sources: {len(chunks)} chunks')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def create_citation_bibliography(self, citations: List[str], chunks) -> str:
        """Create a formatted bibliography from citations."""
        bibliography = []
        
        for citation_id in citations:
            citation_info = self._parse_citation_id(citation_id)
            paper_info = self._extract_paper_info(chunks, citation_info['id'])
            
            # Format as academic citation
            bib_entry = []
            if paper_info['authors'] != "Unknown Authors":
                bib_entry.append(paper_info['authors'])
            if paper_info['title'] != "Unknown Title":
                bib_entry.append(f'"{paper_info["title"]}"')
            if paper_info['year']:
                bib_entry.append(f"({paper_info['year']})")
            if paper_info['arxiv_id']:
                bib_entry.append(f"arXiv:{paper_info['arxiv_id']}")
            elif paper_info['doi']:
                bib_entry.append(f"DOI:{paper_info['doi']}")
            
            bibliography.append(". ".join(bib_entry))
        
        return "\n".join(f"{i+1}. {entry}" for i, entry in enumerate(bibliography))
    
    def get_citation_count(self, answer: str) -> int:
        """Count the number of citations in an answer."""
        citation_pattern = r'\[[^\]]+\]'
        citations = re.findall(citation_pattern, answer)
        return len(set(citations))  # Unique citations only
    
    def validate_citations(self, answer: str, available_citations: List[str]) -> Dict[str, Any]:
        """Validate that all citations in answer are available."""
        cited_in_answer = re.findall(r'\[([^\]]+)\]', answer)
        
        missing_citations = []
        valid_citations = []
        
        for citation in cited_in_answer:
            if citation in available_citations:
                valid_citations.append(citation)
            else:
                missing_citations.append(citation)
        
        return {
            "valid_citations": valid_citations,
            "missing_citations": missing_citations,
            "citation_coverage": len(valid_citations) / len(cited_in_answer) if cited_in_answer else 1.0
        }

if __name__ == "__main__":
    # Example usage
    formatter = AnswerFormatter(citation_style="academic")
    
    # Mock data for testing
    class MockChunk:
        def __init__(self, paper_id, metadata):
            self.paper_id = paper_id
            self.metadata = metadata
    
    chunks = [
        MockChunk("paper1", {
            "title": "Attention Is All You Need",
            "authors": "Vaswani, A., et al.",
            "arxiv_id": "1706.03762",
            "year": 2017
        })
    ]
    
    answer = "The transformer architecture [arxiv:1706.03762] revolutionized natural language processing."
    citations = ["arxiv:1706.03762"]
    
    # Test different formats
    print("=== JSON Format ===")
    print(formatter.format_as_json(answer, citations, chunks, query="What is transformer?"))
    
    print("\n=== Markdown Format ===")
    print(formatter.format_as_markdown(answer, citations, chunks, query="What is transformer?"))
    
    print("\n=== HTML Format ===")
    print(formatter.format_as_html(answer, citations, chunks, query="What is transformer?"))
