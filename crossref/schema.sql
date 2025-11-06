-- Cross-Reference Database Schema
-- Stores relationships between research papers

-- Table for storing cross-reference relationships
CREATE TABLE IF NOT EXISTS crossref_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_paper TEXT NOT NULL,
    score FLOAT NOT NULL DEFAULT 1.0,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    created_date TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT, -- JSON metadata
    UNIQUE(source_paper, relation, target_paper)
);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_source_paper ON crossref_relationships(source_paper);
CREATE INDEX IF NOT EXISTS idx_target_paper ON crossref_relationships(target_paper);
CREATE INDEX IF NOT EXISTS idx_relation ON crossref_relationships(relation);
CREATE INDEX IF NOT EXISTS idx_score ON crossref_relationships(score);
CREATE INDEX IF NOT EXISTS idx_created_date ON crossref_relationships(created_date);

-- Table for storing paper metadata (denormalized for cross-ref queries)
CREATE TABLE IF NOT EXISTS papers_metadata (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT, -- JSON array
    year INTEGER,
    doi TEXT,
    arxiv_id TEXT,
    abstract TEXT,
    keywords TEXT, -- JSON array
    citation_count INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now')),
    metadata TEXT -- JSON metadata
);

-- Indexes for paper metadata
CREATE INDEX IF NOT EXISTS idx_paper_title ON papers_metadata(title);
CREATE INDEX IF NOT EXISTS idx_paper_year ON papers_metadata(year);
CREATE INDEX IF NOT EXISTS idx_paper_doi ON papers_metadata(doi);
CREATE INDEX IF NOT EXISTS idx_paper_arxiv ON papers_metadata(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_citation_count ON papers_metadata(citation_count);

-- Table for citation details
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citing_paper TEXT NOT NULL,
    cited_paper TEXT,
    raw_citation TEXT,
    extracted_title TEXT,
    extracted_authors TEXT,
    extracted_year INTEGER,
    extracted_doi TEXT,
    extracted_arxiv TEXT,
    match_type TEXT, -- 'doi', 'arxiv', 'title', 'fuzzy'
    match_confidence FLOAT,
    created_date TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (citing_paper) REFERENCES papers_metadata(paper_id)
);

-- Indexes for citations
CREATE INDEX IF NOT EXISTS idx_citing_paper ON citations(citing_paper);
CREATE INDEX IF NOT EXISTS idx_cited_paper ON citations(cited_paper);
CREATE INDEX IF NOT EXISTS idx_citation_doi ON citations(extracted_doi);
CREATE INDEX IF NOT EXISTS idx_citation_arxiv ON citations(extracted_arxiv);
CREATE INDEX IF NOT EXISTS idx_match_type ON citations(match_type);

-- Table for similarity relationships
CREATE TABLE IF NOT EXISTS similarities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper1 TEXT NOT NULL,
    paper2 TEXT NOT NULL,
    similarity_score FLOAT NOT NULL,
    similarity_type TEXT NOT NULL, -- 'semantic', 'title', 'abstract'
    embedding_model TEXT,
    created_date TEXT DEFAULT (datetime('now')),
    metadata TEXT, -- JSON metadata
    CHECK (paper1 < paper2), -- Enforce ordering to prevent duplicates
    UNIQUE(paper1, paper2, similarity_type)
);

-- Indexes for similarities
CREATE INDEX IF NOT EXISTS idx_similarity_paper1 ON similarities(paper1);
CREATE INDEX IF NOT EXISTS idx_similarity_paper2 ON similarities(paper2);
CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarities(similarity_score);
CREATE INDEX IF NOT EXISTS idx_similarity_type ON similarities(similarity_type);

-- Table for topic clusters/communities
CREATE TABLE IF NOT EXISTS topic_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    cluster_method TEXT, -- 'community_detection', 'kmeans', 'hierarchical'
    cluster_score FLOAT,
    created_date TEXT DEFAULT (datetime('now')),
    UNIQUE(cluster_id, paper_id)
);

-- Indexes for topic clusters
CREATE INDEX IF NOT EXISTS idx_cluster_id ON topic_clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_paper ON topic_clusters(paper_id);
CREATE INDEX IF NOT EXISTS idx_cluster_method ON topic_clusters(cluster_method);

-- Table for author relationships
CREATE TABLE IF NOT EXISTS author_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper1 TEXT NOT NULL,
    paper2 TEXT NOT NULL,
    shared_authors TEXT, -- JSON array of shared authors
    author_overlap FLOAT, -- Ratio of shared/total authors
    created_date TEXT DEFAULT (datetime('now')),
    CHECK (paper1 < paper2), -- Enforce ordering
    UNIQUE(paper1, paper2)
);

-- Indexes for author relationships
CREATE INDEX IF NOT EXISTS idx_author_paper1 ON author_relationships(paper1);
CREATE INDEX IF NOT EXISTS idx_author_paper2 ON author_relationships(paper2);
CREATE INDEX IF NOT EXISTS idx_author_overlap ON author_relationships(author_overlap);

-- View for easy querying of all relationships
CREATE VIEW IF NOT EXISTS all_relationships AS
SELECT 
    source_paper,
    target_paper,
    relation,
    score,
    confidence,
    created_date,
    'crossref' as source_table
FROM crossref_relationships

UNION ALL

SELECT 
    paper1 as source_paper,
    paper2 as target_paper,
    'similar_' || similarity_type as relation,
    similarity_score as score,
    similarity_score as confidence,
    created_date,
    'similarities' as source_table
FROM similarities

UNION ALL

SELECT 
    paper1 as source_paper,
    paper2 as target_paper,
    'same_author' as relation,
    author_overlap as score,
    1.0 as confidence,
    created_date,
    'author_relationships' as source_table
FROM author_relationships;

-- View for paper statistics
CREATE VIEW IF NOT EXISTS paper_statistics AS
SELECT 
    p.paper_id,
    p.title,
    p.authors,
    p.year,
    p.citation_count,
    
    -- Citation statistics
    COALESCE(cite_out.outgoing_citations, 0) as outgoing_citations,
    COALESCE(cite_in.incoming_citations, 0) as incoming_citations,
    
    -- Similarity statistics
    COALESCE(sim.similar_papers, 0) as similar_papers,
    COALESCE(sim.avg_similarity, 0.0) as avg_similarity,
    
    -- Author collaboration statistics
    COALESCE(auth.collaborator_papers, 0) as collaborator_papers
    
FROM papers_metadata p

LEFT JOIN (
    SELECT citing_paper, COUNT(*) as outgoing_citations
    FROM citations 
    WHERE cited_paper IS NOT NULL
    GROUP BY citing_paper
) cite_out ON p.paper_id = cite_out.citing_paper

LEFT JOIN (
    SELECT cited_paper, COUNT(*) as incoming_citations
    FROM citations 
    WHERE cited_paper IS NOT NULL
    GROUP BY cited_paper
) cite_in ON p.paper_id = cite_in.cited_paper

LEFT JOIN (
    SELECT 
        paper_id,
        COUNT(*) as similar_papers,
        AVG(similarity_score) as avg_similarity
    FROM (
        SELECT paper1 as paper_id, similarity_score FROM similarities
        UNION ALL
        SELECT paper2 as paper_id, similarity_score FROM similarities
    ) s
    GROUP BY paper_id
) sim ON p.paper_id = sim.paper_id

LEFT JOIN (
    SELECT 
        paper_id,
        COUNT(*) as collaborator_papers
    FROM (
        SELECT paper1 as paper_id FROM author_relationships
        UNION ALL
        SELECT paper2 as paper_id FROM author_relationships
    ) a
    GROUP BY paper_id
) auth ON p.paper_id = auth.paper_id;

-- Triggers to maintain citation counts
CREATE TRIGGER IF NOT EXISTS update_citation_count_insert
AFTER INSERT ON citations
WHEN NEW.cited_paper IS NOT NULL
BEGIN
    UPDATE papers_metadata 
    SET citation_count = citation_count + 1,
        last_updated = datetime('now')
    WHERE paper_id = NEW.cited_paper;
END;

CREATE TRIGGER IF NOT EXISTS update_citation_count_delete
AFTER DELETE ON citations
WHEN OLD.cited_paper IS NOT NULL
BEGIN
    UPDATE papers_metadata 
    SET citation_count = citation_count - 1,
        last_updated = datetime('now')
    WHERE paper_id = OLD.cited_paper;
END;

-- Function-like queries (stored as comments for reference)

-- Find papers similar to a given paper:
/*
SELECT 
    CASE 
        WHEN s.paper1 = ? THEN s.paper2 
        ELSE s.paper1 
    END as related_paper,
    s.similarity_score,
    p.title,
    p.authors
FROM similarities s
JOIN papers_metadata p ON (
    p.paper_id = CASE 
        WHEN s.paper1 = ? THEN s.paper2 
        ELSE s.paper1 
    END
)
WHERE (s.paper1 = ? OR s.paper2 = ?)
    AND s.similarity_score >= ?
ORDER BY s.similarity_score DESC;
*/

-- Find citation chain (papers that cite papers that cite a given paper):
/*
WITH RECURSIVE citation_chain(paper_id, level) AS (
    -- Base case: papers directly citing the target
    SELECT citing_paper, 1
    FROM citations
    WHERE cited_paper = ?
    
    UNION ALL
    
    -- Recursive case: papers citing papers in the chain
    SELECT c.citing_paper, cc.level + 1
    FROM citations c
    JOIN citation_chain cc ON c.cited_paper = cc.paper_id
    WHERE cc.level < 3  -- Limit depth
)
SELECT 
    cc.paper_id,
    cc.level,
    p.title,
    p.authors,
    p.year
FROM citation_chain cc
JOIN papers_metadata p ON cc.paper_id = p.paper_id
ORDER BY cc.level, p.year DESC;
*/

-- Find influential papers in a topic cluster:
/*
SELECT 
    p.paper_id,
    p.title,
    p.authors,
    p.year,
    p.citation_count,
    COUNT(DISTINCT c.citing_paper) as direct_citations,
    AVG(s.similarity_score) as avg_cluster_similarity
FROM papers_metadata p
JOIN topic_clusters tc ON p.paper_id = tc.paper_id
LEFT JOIN citations c ON p.paper_id = c.cited_paper
LEFT JOIN similarities s ON (p.paper_id = s.paper1 OR p.paper_id = s.paper2)
WHERE tc.cluster_id = ?
GROUP BY p.paper_id
ORDER BY p.citation_count DESC, direct_citations DESC
LIMIT 10;
*/
