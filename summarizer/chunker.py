"""
Text Chunker

Intelligent text chunking for research papers using LangChain.
Supports section-aware splitting, token-based limits, and semantic boundaries.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
    import tiktoken
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
        import tiktoken
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        RecursiveCharacterTextSplitter = None
        TokenTextSplitter = None
        tiktoken = None
        LANGCHAIN_AVAILABLE = False

from .pdf_extractor import Section, PaperStructure

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Different chunking strategies."""
    SECTION_AWARE = "section_aware"  # Respect section boundaries
    TOKEN_BASED = "token_based"      # Fixed token limits
    SEMANTIC = "semantic"            # Semantic boundary detection
    HYBRID = "hybrid"                # Combination approach


@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""
    chunk_id: str
    source_file: str
    section_title: str
    chunk_index: int
    total_chunks: int
    token_count: int
    char_count: int
    overlap_tokens: int
    chunk_type: str  # 'title', 'abstract', 'section', 'references'
    page_numbers: List[int]


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for storage."""
        return {
            'content': self.content,
            'metadata': {
                'chunk_id': self.metadata.chunk_id,
                'source_file': self.metadata.source_file,
                'section_title': self.metadata.section_title,
                'chunk_index': self.metadata.chunk_index,
                'total_chunks': self.metadata.total_chunks,
                'token_count': self.metadata.token_count,
                'char_count': self.metadata.char_count,
                'overlap_tokens': self.metadata.overlap_tokens,
                'chunk_type': self.metadata.chunk_type,
                'page_numbers': self.metadata.page_numbers
            }
        }


class ResearchPaperChunker:
    """
    Intelligent chunking for research papers.
    
    Handles different content types (title, abstract, sections, references)
    with appropriate chunking strategies for each.
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        max_chunk_tokens: int = 1000,
        overlap_tokens: int = 100,
        min_chunk_tokens: int = 50,
        model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the chunker.
        
        Args:
            strategy: Chunking strategy to use
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between consecutive chunks
            min_chunk_tokens: Minimum tokens for a valid chunk
            model_name: Model name for token counting
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, using fallback text splitters")
            self._use_fallback_splitters = True
        else:
            self._use_fallback_splitters = False
        
        self.strategy = strategy
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.model_name = model_name
        
        # Initialize tokenizer
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except Exception:
                try:
                    logger.warning(f"Could not load tokenizer for {model_name}, using cl100k_base")
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    logger.warning("Tiktoken not available, using character-based estimation")
                    self.tokenizer = None
        else:
            logger.warning("Tiktoken not available, using character-based estimation")
            self.tokenizer = None
        
        # Initialize text splitters
        self._init_splitters()
    
    def _init_splitters(self):
        """Initialize different text splitters."""
        if self._use_fallback_splitters:
            # Use simple fallback splitters
            self.token_splitter = None
            self.char_splitter = None
        else:
            # Token-based splitter
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.max_chunk_tokens,
                chunk_overlap=self.overlap_tokens,
                model_name=self.model_name
            )
            
            # Character-based splitter for section awareness
            self.char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_tokens * 4,  # Rough char to token ratio
                chunk_overlap=self.overlap_tokens * 4,
                separators=[
                    "\n\n\n",  # Major breaks
                    "\n\n",    # Paragraph breaks
                    "\n",      # Line breaks  
                    ". ",      # Sentence breaks
                    " ",       # Word breaks
                    ""         # Character breaks
                ],
                length_function=self._count_tokens
            )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback to character-based estimation
        return len(text) // 4
    
    def chunk_paper(self, paper_structure: PaperStructure) -> List[TextChunk]:
        """
        Chunk an entire research paper.
        
        Args:
            paper_structure: Extracted paper structure
            
        Returns:
            List of text chunks with metadata
        """
        logger.info(f"Chunking paper with {len(paper_structure.sections)} sections using {self.strategy.value} strategy")
        
        chunks = []
        chunk_counter = 0
        
        # Get source file path
        source_file = paper_structure.metadata.get('file_path', 'unknown')
        
        # Chunk title (if available)
        if paper_structure.title.strip():
            title_chunks = self._chunk_title(
                paper_structure.title,
                source_file,
                chunk_counter
            )
            chunks.extend(title_chunks)
            chunk_counter += len(title_chunks)
        
        # Chunk abstract (if available)
        if paper_structure.abstract.strip():
            abstract_chunks = self._chunk_abstract(
                paper_structure.abstract,
                source_file,
                chunk_counter
            )
            chunks.extend(abstract_chunks)
            chunk_counter += len(abstract_chunks)
        
        # Chunk sections
        for section in paper_structure.sections:
            if not section.content.strip():
                continue
                
            section_chunks = self._chunk_section(
                section,
                source_file,
                chunk_counter
            )
            chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)
        
        # Chunk references (if available)
        if paper_structure.references.strip():
            ref_chunks = self._chunk_references(
                paper_structure.references,
                source_file,
                chunk_counter
            )
            chunks.extend(ref_chunks)
            chunk_counter += len(ref_chunks)
        
        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from paper")
        return chunks
    
    def _chunk_title(self, title: str, source_file: str, start_index: int) -> List[TextChunk]:
        """Chunk paper title (usually fits in one chunk)."""
        token_count = self._count_tokens(title)
        
        chunk = TextChunk(
            content=title.strip(),
            metadata=ChunkMetadata(
                chunk_id=f"{source_file}_title_0",
                source_file=source_file,
                section_title="Title",
                chunk_index=start_index,
                total_chunks=0,  # Will be updated later
                token_count=token_count,
                char_count=len(title),
                overlap_tokens=0,
                chunk_type="title",
                page_numbers=[0]
            )
        )
        
        return [chunk]
    
    def _chunk_abstract(self, abstract: str, source_file: str, start_index: int) -> List[TextChunk]:
        """Chunk paper abstract."""
        token_count = self._count_tokens(abstract)
        
        # Abstract usually fits in one chunk, but split if too large
        if token_count <= self.max_chunk_tokens:
            chunk = TextChunk(
                content=abstract.strip(),
                metadata=ChunkMetadata(
                    chunk_id=f"{source_file}_abstract_0",
                    source_file=source_file,
                    section_title="Abstract",
                    chunk_index=start_index,
                    total_chunks=0,
                    token_count=token_count,
                    char_count=len(abstract),
                    overlap_tokens=0,
                    chunk_type="abstract",
                    page_numbers=[0, 1]
                )
            )
            return [chunk]
        else:
            # Split large abstract
            return self._split_text_with_metadata(
                text=abstract,
                section_title="Abstract",
                source_file=source_file,
                chunk_type="abstract",
                start_index=start_index,
                page_numbers=[0, 1]
            )
    
    def _chunk_section(self, section: Section, source_file: str, start_index: int) -> List[TextChunk]:
        """Chunk a paper section."""
        if not section.content.strip():
            return []
        
        page_numbers = list(range(section.page_start, section.page_end + 1))
        
        # Choose chunking approach based on strategy
        if self.strategy == ChunkingStrategy.SECTION_AWARE:
            return self._chunk_section_aware(section, source_file, start_index, page_numbers)
        elif self.strategy == ChunkingStrategy.TOKEN_BASED:
            return self._chunk_token_based(section, source_file, start_index, page_numbers)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(section, source_file, start_index, page_numbers)
        else:  # HYBRID
            return self._chunk_hybrid(section, source_file, start_index, page_numbers)
    
    def _chunk_section_aware(self, section: Section, source_file: str, start_index: int, page_numbers: List[int]) -> List[TextChunk]:
        """Chunk respecting section boundaries."""
        # Try to keep entire section in one chunk if possible
        token_count = self._count_tokens(section.content)
        
        if token_count <= self.max_chunk_tokens:
            chunk = TextChunk(
                content=section.content.strip(),
                metadata=ChunkMetadata(
                    chunk_id=f"{source_file}_{self._sanitize_title(section.title)}_0",
                    source_file=source_file,
                    section_title=section.title,
                    chunk_index=start_index,
                    total_chunks=0,
                    token_count=token_count,
                    char_count=len(section.content),
                    overlap_tokens=0,
                    chunk_type="section",
                    page_numbers=page_numbers
                )
            )
            return [chunk]
        else:
            # Split section while respecting paragraph boundaries
            return self._split_text_with_metadata(
                text=section.content,
                section_title=section.title,
                source_file=source_file,
                chunk_type="section",
                start_index=start_index,
                page_numbers=page_numbers
            )
    
    def _chunk_token_based(self, section: Section, source_file: str, start_index: int, page_numbers: List[int]) -> List[TextChunk]:
        """Chunk based on token limits only."""
        return self._split_text_with_metadata(
            text=section.content,
            section_title=section.title,
            source_file=source_file,
            chunk_type="section",
            start_index=start_index,
            page_numbers=page_numbers,
            use_token_splitter=True
        )
    
    def _chunk_semantic(self, section: Section, source_file: str, start_index: int, page_numbers: List[int]) -> List[TextChunk]:
        """Chunk based on semantic boundaries."""
        # For now, use the character splitter with semantic separators
        # In the future, this could use more sophisticated semantic analysis
        return self._split_text_with_metadata(
            text=section.content,
            section_title=section.title,
            source_file=source_file,
            chunk_type="section",
            start_index=start_index,
            page_numbers=page_numbers,
            use_semantic_separators=True
        )
    
    def _chunk_hybrid(self, section: Section, source_file: str, start_index: int, page_numbers: List[int]) -> List[TextChunk]:
        """Hybrid approach combining multiple strategies."""
        # Try section-aware first, fall back to token-based if needed
        section_chunks = self._chunk_section_aware(section, source_file, start_index, page_numbers)
        
        # Check if any chunks are still too large
        refined_chunks = []
        chunk_index_offset = 0
        
        for chunk in section_chunks:
            if chunk.metadata.token_count > self.max_chunk_tokens:
                # Further split this chunk
                sub_chunks = self._split_text_with_metadata(
                    text=chunk.content,
                    section_title=chunk.metadata.section_title,
                    source_file=source_file,
                    chunk_type="section",
                    start_index=start_index + chunk_index_offset,
                    page_numbers=page_numbers,
                    use_token_splitter=True
                )
                refined_chunks.extend(sub_chunks)
                chunk_index_offset += len(sub_chunks)
            else:
                # Update chunk index
                chunk.metadata.chunk_index = start_index + chunk_index_offset
                refined_chunks.append(chunk)
                chunk_index_offset += 1
        
        return refined_chunks
    
    def _chunk_references(self, references: str, source_file: str, start_index: int) -> List[TextChunk]:
        """Chunk references section."""
        return self._split_text_with_metadata(
            text=references,
            section_title="References",
            source_file=source_file,
            chunk_type="references",
            start_index=start_index,
            page_numbers=[-1]  # Usually at the end
        )
    
    def _split_text_with_metadata(
        self,
        text: str,
        section_title: str,
        source_file: str,
        chunk_type: str,
        start_index: int,
        page_numbers: List[int],
        use_token_splitter: bool = False,
        use_semantic_separators: bool = False
    ) -> List[TextChunk]:
        """Split text and create chunks with metadata."""
        
        if self._use_fallback_splitters:
            # Use simple fallback splitting
            text_chunks = self._fallback_split_text(text)
        else:
            # Choose splitter
            if use_token_splitter and self.token_splitter:
                text_chunks = self.token_splitter.split_text(text)
            elif self.char_splitter:
                text_chunks = self.char_splitter.split_text(text)
            else:
                text_chunks = self._fallback_split_text(text)
        
        # Filter out chunks that are too small
        text_chunks = [chunk for chunk in text_chunks 
                      if self._count_tokens(chunk) >= self.min_chunk_tokens]
        
        if not text_chunks:
            return []
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{source_file}_{self._sanitize_title(section_title)}_{i}"
            token_count = self._count_tokens(chunk_text)
            
            # Calculate overlap tokens
            overlap_tokens = 0
            if i > 0:
                # Estimate overlap with previous chunk
                overlap_tokens = min(self.overlap_tokens, token_count // 4)
            
            chunk = TextChunk(
                content=chunk_text.strip(),
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    source_file=source_file,
                    section_title=section_title,
                    chunk_index=start_index + i,
                    total_chunks=0,  # Will be updated later
                    token_count=token_count,
                    char_count=len(chunk_text),
                    overlap_tokens=overlap_tokens,
                    chunk_type=chunk_type,
                    page_numbers=page_numbers
                )
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_split_text(self, text: str) -> List[str]:
        """Simple fallback text splitting when LangChain is not available."""
        # Simple splitting based on separators
        separators = ['\n\n\n', '\n\n', '\n', '. ', ' ']
        chunks = [text]
        
        for separator in separators:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.max_chunk_tokens * 4:  # Rough char estimate
                    new_chunks.append(chunk)
                else:
                    # Split by separator
                    parts = chunk.split(separator)
                    current_chunk = ""
                    
                    for part in parts:
                        test_chunk = current_chunk + separator + part if current_chunk else part
                        if len(test_chunk) <= self.max_chunk_tokens * 4:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                new_chunks.append(current_chunk)
                            current_chunk = part
                    
                    if current_chunk:
                        new_chunks.append(current_chunk)
            
            chunks = new_chunks
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _sanitize_title(self, title: str) -> str:
        """Sanitize section title for use in chunk IDs."""
        # Remove special characters and spaces
        sanitized = re.sub(r'[^\w\s-]', '', title.lower())
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized[:50]  # Limit length
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        token_counts = [chunk.metadata.token_count for chunk in chunks]
        char_counts = [chunk.metadata.char_count for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'total_chars': sum(char_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks),
            'avg_chars_per_chunk': sum(char_counts) / len(chunks),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'chunk_types': {chunk_type: len([c for c in chunks if c.metadata.chunk_type == chunk_type])
                           for chunk_type in set(chunk.metadata.chunk_type for chunk in chunks)}
        }


def chunk_paper_structure(
    paper_structure: PaperStructure,
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
    max_chunk_tokens: int = 1000,
    overlap_tokens: int = 100
) -> List[TextChunk]:
    """
    Convenience function to chunk a paper structure.
    
    Args:
        paper_structure: Extracted paper structure
        strategy: Chunking strategy
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunker = ResearchPaperChunker(
        strategy=strategy,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens
    )
    
    return chunker.chunk_paper(paper_structure)


if __name__ == "__main__":
    # Example usage
    import sys
    import json
    from .pdf_extractor import extract_pdf_structure
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <pdf_path> [max_tokens] [strategy]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    strategy_name = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
    
    # Map strategy name
    strategy_map = {
        'section': ChunkingStrategy.SECTION_AWARE,
        'token': ChunkingStrategy.TOKEN_BASED,
        'semantic': ChunkingStrategy.SEMANTIC,
        'hybrid': ChunkingStrategy.HYBRID
    }
    strategy = strategy_map.get(strategy_name.lower(), ChunkingStrategy.HYBRID)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Extract paper structure
        paper_structure = extract_pdf_structure(pdf_path)
        
        # Chunk the paper
        chunks = chunk_paper_structure(
            paper_structure,
            strategy=strategy,
            max_chunk_tokens=max_tokens
        )
        
        # Get stats
        chunker = ResearchPaperChunker(strategy=strategy, max_chunk_tokens=max_tokens)
        stats = chunker.get_chunk_stats(chunks)
        
        print(f"Chunked paper: {pdf_path}")
        print(f"Strategy: {strategy.value}")
        print(f"Stats: {json.dumps(stats, indent=2)}")
        
        # Show first few chunks
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} ({chunk.metadata.chunk_type}):")
            print(f"Section: {chunk.metadata.section_title}")
            print(f"Tokens: {chunk.metadata.token_count}")
            print(f"Content preview: {chunk.content[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
