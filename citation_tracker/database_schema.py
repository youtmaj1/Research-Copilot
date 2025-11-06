"""
Database Schema Module

This module defines and manages the database schema for citation tracking,
including tables for citations, citation relationships, temporal data, and
integration with the existing papers.db database.

Key Features:
- Define citation tracking tables
- Create indexes for performance
- Handle database migrations
- Integrate with existing papers.db schema
- Support for temporal citation analysis
- Backup and restore capabilities

Functions:
    create_citation_tables: Create all citation tracking tables
    create_indexes: Create performance indexes
    migrate_database: Handle schema migrations
    backup_database: Create database backup
    restore_database: Restore from backup
"""

import logging
import sqlite3
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database schema version
SCHEMA_VERSION = "1.0.0"

def create_citation_tables(db_path: str = "papers.db") -> bool:
    """
    Create all citation tracking tables in the database.
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        bool: True if tables created successfully
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        logger.info(f"Creating citation tracking tables in {db_path}")
        
        # Table 1: Extracted Citations
        # Stores raw citations extracted from papers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                doi TEXT,
                arxiv_id TEXT,
                title TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                normalized_text TEXT,
                confidence REAL DEFAULT 0.0,
                extraction_method TEXT DEFAULT 'unknown',
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)
        
        # Table 2: Citation Matches
        # Stores resolved matches between extracted citations and known papers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extracted_citation_id INTEGER NOT NULL,
                citing_paper_id TEXT NOT NULL,
                cited_paper_id TEXT NOT NULL,
                match_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                paper_title TEXT,
                paper_authors TEXT,
                paper_year INTEGER,
                paper_doi TEXT,
                paper_arxiv_id TEXT,
                is_ambiguous BOOLEAN DEFAULT FALSE,
                alternative_matches TEXT,  -- JSON array of alternative paper IDs
                matched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (extracted_citation_id) REFERENCES extracted_citations (id) ON DELETE CASCADE,
                FOREIGN KEY (citing_paper_id) REFERENCES papers (id) ON DELETE CASCADE,
                FOREIGN KEY (cited_paper_id) REFERENCES papers (id) ON DELETE CASCADE,
                UNIQUE (citing_paper_id, cited_paper_id)
            )
        """)
        
        # Table 3: Citation Graph Nodes
        # Stores paper nodes with graph metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_nodes (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                doi TEXT,
                arxiv_id TEXT,
                venue TEXT,
                citation_count INTEGER DEFAULT 0,
                reference_count INTEGER DEFAULT 0,
                h_index REAL DEFAULT 0.0,
                pagerank REAL DEFAULT 0.0,
                betweenness_centrality REAL DEFAULT 0.0,
                closeness_centrality REAL DEFAULT 0.0,
                clustering_coefficient REAL DEFAULT 0.0,
                added_to_graph_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_metrics_update TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)
        
        # Table 4: Citation Graph Edges
        # Stores citation relationships between papers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citing_paper_id TEXT NOT NULL,
                cited_paper_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                match_type TEXT DEFAULT 'unknown',
                extraction_method TEXT DEFAULT 'unknown',
                weight REAL DEFAULT 1.0,
                context TEXT,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (citing_paper_id) REFERENCES citation_nodes (paper_id) ON DELETE CASCADE,
                FOREIGN KEY (cited_paper_id) REFERENCES citation_nodes (paper_id) ON DELETE CASCADE,
                UNIQUE (citing_paper_id, cited_paper_id)
            )
        """)
        
        # Table 5: Citation History (for temporal analysis)
        # Stores citation count snapshots over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                citation_count INTEGER NOT NULL,
                new_citations INTEGER DEFAULT 0,
                cumulative_citations INTEGER DEFAULT 0,
                citation_rate REAL DEFAULT 0.0,
                acceleration REAL DEFAULT 0.0,
                measurement_period_days INTEGER DEFAULT 0,
                data_source TEXT DEFAULT 'manual',
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)
        
        # Table 6: Trending Papers
        # Stores papers with temporal trend information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trending_papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                current_citations INTEGER NOT NULL,
                previous_citations INTEGER NOT NULL,
                citation_growth INTEGER NOT NULL,
                growth_rate REAL NOT NULL,
                trend_score REAL NOT NULL,
                velocity REAL NOT NULL,
                acceleration REAL NOT NULL,
                trend_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                time_window_days INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)
        
        # Table 7: Citation Bursts
        # Stores detected citation burst events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_bursts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                burst_start TEXT NOT NULL,
                burst_end TEXT,
                peak_timestamp TEXT NOT NULL,
                peak_citation_rate REAL NOT NULL,
                z_score REAL NOT NULL,
                burst_intensity REAL NOT NULL,
                total_burst_citations INTEGER DEFAULT 0,
                detection_method TEXT DEFAULT 'statistical',
                detected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)
        
        # Table 8: Graph Snapshots
        # Stores graph metadata and snapshots for versioning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_name TEXT UNIQUE NOT NULL,
                description TEXT,
                node_count INTEGER NOT NULL,
                edge_count INTEGER NOT NULL,
                creation_date TEXT NOT NULL,
                metrics TEXT,  -- JSON object with graph metrics
                file_path TEXT,  -- Path to exported graph file
                format TEXT DEFAULT 'json',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 9: Citation Processing Log
        # Stores processing history and errors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                operation TEXT NOT NULL,  -- extract, resolve, graph_add, etc.
                status TEXT NOT NULL,     -- success, error, warning
                message TEXT,
                details TEXT,             -- JSON object with additional details
                processing_time REAL,     -- Processing time in seconds
                processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)
        
        # Table 10: Schema Version and Metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert schema version
        cursor.execute("""
            INSERT OR REPLACE INTO citation_schema_info (key, value)
            VALUES ('version', ?), ('created_at', ?), ('description', ?)
        """, (SCHEMA_VERSION, datetime.now().isoformat(), 
              'Citation Tracker Database Schema'))
        
        connection.commit()
        logger.info("Citation tracking tables created successfully")
        
        # Create indexes for performance
        success = _create_indexes_internal(connection, cursor)
        if not success:
            logger.warning("Failed to create some indexes")
        
        connection.close()
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error creating citation tables: {e}")
        return False

def _create_indexes_internal(connection: sqlite3.Connection, cursor: sqlite3.Cursor) -> bool:
    """
    Create performance indexes on citation tracking tables.
    
    Args:
        connection: Database connection
        cursor: Database cursor
        
    Returns:
        bool: True if indexes created successfully
    """
    try:
        logger.info("Creating performance indexes")
        
        # Indexes for extracted_citations table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_extracted_citations_source_paper ON extracted_citations (source_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_extracted_citations_doi ON extracted_citations (doi) WHERE doi IS NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_extracted_citations_arxiv ON extracted_citations (arxiv_id) WHERE arxiv_id IS NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_extracted_citations_year ON extracted_citations (year) WHERE year IS NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_extracted_citations_extracted_at ON extracted_citations (extracted_at)")
        
        # Indexes for citation_matches table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_matches_citing_paper ON citation_matches (citing_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_matches_cited_paper ON citation_matches (cited_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_matches_match_type ON citation_matches (match_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_matches_confidence ON citation_matches (confidence DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_matches_matched_at ON citation_matches (matched_at)")
        
        # Indexes for citation_nodes table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_nodes_citation_count ON citation_nodes (citation_count DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_nodes_pagerank ON citation_nodes (pagerank DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_nodes_year ON citation_nodes (year) WHERE year IS NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_nodes_venue ON citation_nodes (venue) WHERE venue IS NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_nodes_doi ON citation_nodes (doi) WHERE doi IS NOT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_nodes_arxiv ON citation_nodes (arxiv_id) WHERE arxiv_id IS NOT NULL")
        
        # Indexes for citation_edges table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_edges_citing_paper ON citation_edges (citing_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_edges_cited_paper ON citation_edges (cited_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_edges_confidence ON citation_edges (confidence DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_edges_weight ON citation_edges (weight DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_edges_added_at ON citation_edges (added_at)")
        
        # Indexes for citation_history table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_history_paper_timestamp ON citation_history (paper_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_history_timestamp ON citation_history (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_history_citation_rate ON citation_history (citation_rate DESC)")
        
        # Indexes for trending_papers table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trending_papers_trend_score ON trending_papers (trend_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trending_papers_growth_rate ON trending_papers (growth_rate DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trending_papers_trend_type ON trending_papers (trend_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trending_papers_last_updated ON trending_papers (last_updated)")
        
        # Indexes for citation_bursts table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_bursts_paper_id ON citation_bursts (paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_bursts_burst_start ON citation_bursts (burst_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_bursts_intensity ON citation_bursts (burst_intensity DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_bursts_z_score ON citation_bursts (z_score DESC)")
        
        # Indexes for graph_snapshots table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_snapshots_creation_date ON graph_snapshots (creation_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_snapshots_node_count ON graph_snapshots (node_count DESC)")
        
        # Indexes for citation_processing_log table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_log_paper_id ON citation_processing_log (paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_log_operation ON citation_processing_log (operation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_log_status ON citation_processing_log (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_log_processed_at ON citation_processing_log (processed_at)")
        
        connection.commit()
        logger.info("Performance indexes created successfully")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error creating indexes: {e}")
        return False

def verify_schema(db_path: str = "papers.db") -> Dict[str, Any]:
    """
    Verify the citation tracking schema and return status information.
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        Dict[str, Any]: Schema verification results
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Check if citation tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'citation_%' OR name = 'extracted_citations' OR name = 'trending_papers'
        """)
        
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'extracted_citations',
            'citation_matches', 
            'citation_nodes',
            'citation_edges',
            'citation_history',
            'trending_papers',
            'citation_bursts',
            'graph_snapshots',
            'citation_processing_log',
            'citation_schema_info'
        ]
        
        missing_tables = [table for table in expected_tables if table not in existing_tables]
        
        # Get schema version
        schema_version = None
        try:
            cursor.execute("SELECT value FROM citation_schema_info WHERE key = 'version'")
            result = cursor.fetchone()
            if result:
                schema_version = result[0]
        except sqlite3.Error:
            pass
        
        # Get table statistics
        table_stats = {}
        for table in existing_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_stats[table] = count
            except sqlite3.Error:
                table_stats[table] = "error"
        
        connection.close()
        
        return {
            'schema_exists': len(missing_tables) == 0,
            'schema_version': schema_version,
            'expected_version': SCHEMA_VERSION,
            'existing_tables': existing_tables,
            'missing_tables': missing_tables,
            'table_statistics': table_stats,
            'total_tables': len(existing_tables),
            'verification_timestamp': datetime.now().isoformat()
        }
        
    except sqlite3.Error as e:
        logger.error(f"Error verifying schema: {e}")
        return {'error': str(e)}

def migrate_database(db_path: str = "papers.db", target_version: str = SCHEMA_VERSION) -> bool:
    """
    Migrate database schema to target version.
    
    Args:
        db_path (str): Path to the database file
        target_version (str): Target schema version
        
    Returns:
        bool: True if migration successful
    """
    try:
        # For now, just ensure all tables exist
        # In the future, this would handle version-specific migrations
        logger.info(f"Migrating database schema to version {target_version}")
        
        # Check current schema
        schema_info = verify_schema(db_path)
        
        if schema_info.get('schema_exists', False):
            logger.info("Schema already exists and is up to date")
            return True
        
        # Create missing tables
        if schema_info.get('missing_tables'):
            logger.info(f"Creating missing tables: {schema_info['missing_tables']}")
            return create_citation_tables(db_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error migrating database: {e}")
        return False

def backup_database(db_path: str = "papers.db", backup_dir: str = "backups") -> Optional[str]:
    """
    Create a backup of the database.
    
    Args:
        db_path (str): Path to the database file
        backup_dir (str): Directory to store backups
        
    Returns:
        Optional[str]: Path to backup file if successful
    """
    try:
        # Create backup directory
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = Path(db_path).stem
        backup_filename = f"{db_name}_backup_{timestamp}.db"
        backup_filepath = backup_path / backup_filename
        
        # Copy database file
        shutil.copy2(db_path, backup_filepath)
        
        logger.info(f"Database backed up to: {backup_filepath}")
        return str(backup_filepath)
        
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return None

def restore_database(backup_path: str, target_path: str = "papers.db") -> bool:
    """
    Restore database from backup.
    
    Args:
        backup_path (str): Path to backup file
        target_path (str): Path where to restore the database
        
    Returns:
        bool: True if restore successful
    """
    try:
        if not Path(backup_path).exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        # Create backup of current database if it exists
        if Path(target_path).exists():
            current_backup = backup_database(target_path, "restore_backups")
            if current_backup:
                logger.info(f"Current database backed up to: {current_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, target_path)
        
        logger.info(f"Database restored from: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error restoring database: {e}")
        return False

def get_database_statistics(db_path: str = "papers.db") -> Dict[str, Any]:
    """
    Get comprehensive database statistics.
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        Dict[str, Any]: Database statistics
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        stats = {
            'database_path': db_path,
            'database_size_mb': Path(db_path).stat().st_size / (1024 * 1024) if Path(db_path).exists() else 0,
            'tables': {},
            'schema_info': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Get schema information
        schema_info = verify_schema(db_path)
        stats['schema_info'] = schema_info
        
        # Get detailed table statistics
        for table in schema_info.get('existing_tables', []):
            try:
                # Row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Column information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                stats['tables'][table] = {
                    'row_count': row_count,
                    'column_count': len(columns),
                    'columns': [col[1] for col in columns]  # Column names
                }
                
            except sqlite3.Error as e:
                stats['tables'][table] = {'error': str(e)}
        
        connection.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")
        return {'error': str(e)}

def cleanup_old_data(db_path: str = "papers.db", days_to_keep: int = 365) -> Dict[str, int]:
    """
    Clean up old data from temporal tables.
    
    Args:
        db_path (str): Path to the database file
        days_to_keep (int): Number of days of data to keep
        
    Returns:
        Dict[str, int]: Number of records deleted from each table
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        deleted_counts = {}
        
        # Clean up citation_history
        cursor.execute("DELETE FROM citation_history WHERE timestamp < ?", (cutoff_date,))
        deleted_counts['citation_history'] = cursor.rowcount
        
        # Clean up citation_processing_log
        cursor.execute("DELETE FROM citation_processing_log WHERE processed_at < ?", (cutoff_date,))
        deleted_counts['citation_processing_log'] = cursor.rowcount
        
        # Clean up old graph snapshots (keep last 10)
        cursor.execute("""
            DELETE FROM graph_snapshots 
            WHERE id NOT IN (
                SELECT id FROM graph_snapshots 
                ORDER BY creation_date DESC 
                LIMIT 10
            )
        """)
        deleted_counts['graph_snapshots'] = cursor.rowcount
        
        connection.commit()
        connection.close()
        
        total_deleted = sum(deleted_counts.values())
        logger.info(f"Cleaned up {total_deleted} old records")
        
        return deleted_counts
        
    except sqlite3.Error as e:
        logger.error(f"Error cleaning up old data: {e}")
        return {}

def export_schema_ddl(db_path: str = "papers.db", output_file: str = "citation_schema.sql") -> bool:
    """
    Export the database schema as DDL statements.
    
    Args:
        db_path (str): Path to the database file
        output_file (str): Output file for DDL statements
        
    Returns:
        bool: True if export successful
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Get all citation-related table definitions
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND (
                name LIKE 'citation_%' OR 
                name = 'extracted_citations' OR 
                name = 'trending_papers' OR
                name = 'graph_snapshots'
            )
            ORDER BY name
        """)
        
        table_ddls = [row[0] for row in cursor.fetchall() if row[0]]
        
        # Get all citation-related index definitions
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND (
                name LIKE 'idx_%citation%' OR
                name LIKE 'idx_%trending%' OR
                name LIKE 'idx_%graph%' OR
                name LIKE 'idx_%extracted%'
            )
            ORDER BY name
        """)
        
        index_ddls = [row[0] for row in cursor.fetchall() if row[0]]
        
        connection.close()
        
        # Write DDL to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"-- Citation Tracker Database Schema\n")
            f.write(f"-- Generated on: {datetime.now().isoformat()}\n")
            f.write(f"-- Schema Version: {SCHEMA_VERSION}\n\n")
            
            f.write("-- Tables\n")
            for ddl in table_ddls:
                f.write(f"{ddl};\n\n")
            
            f.write("-- Indexes\n")
            for ddl in index_ddls:
                f.write(f"{ddl};\n\n")
        
        logger.info(f"Schema DDL exported to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting schema DDL: {e}")
        return False

# Additional utility functions for testing compatibility

def optimize_database(db_path: str) -> bool:
    """
    Optimize database by running VACUUM and ANALYZE.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if optimization succeeded, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Run VACUUM to reclaim space and defragment
        cursor.execute("VACUUM")
        
        # Run ANALYZE to update query planner statistics
        cursor.execute("ANALYZE")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        return False

def migrate_schema(db_path: str, target_version: int = 1) -> bool:
    """
    Migrate database schema to target version.
    
    Args:
        db_path: Path to the database file
        target_version: Target schema version
        
    Returns:
        bool: True if migration succeeded, False otherwise
    """
    try:
        # For now, just ensure the current schema exists
        return create_citation_tables(db_path)
    except Exception as e:
        logger.error(f"Error migrating schema: {e}")
        return False

def get_table_info(db_path: str, table_name: str) -> Optional[dict]:
    """
    Get detailed information about a table.
    
    Args:
        db_path: Path to the database file
        table_name: Name of the table
        
    Returns:
        dict: Table information with columns and indexes
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                'name': row[1],
                'type': row[2],
                'notnull': bool(row[3]),
                'default': row[4],
                'pk': bool(row[5])
            })
        
        # Get index information
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'columns': columns,
            'indexes': indexes
        }
        
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return None

def create_indexes(db_path: str) -> bool:
    """
    Create performance indexes on citation tables.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if indexes were created successfully
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create indexes for extracted_citations
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extracted_citations_source 
            ON extracted_citations(source_paper_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extracted_citations_doi 
            ON extracted_citations(doi)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extracted_citations_created 
            ON extracted_citations(created_at)
        """)
        
        # Create indexes for citation_matches
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_matches_citation 
            ON citation_matches(citation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_matches_paper 
            ON citation_matches(paper_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_matches_confidence 
            ON citation_matches(confidence)
        """)
        
        # Create indexes for citation_snapshots
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_snapshots_paper 
            ON citation_snapshots(paper_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_snapshots_timestamp 
            ON citation_snapshots(timestamp)
        """)
        
        # Create indexes for citation_edges
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_edges_citing 
            ON citation_edges(citing_paper_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation_edges_cited 
            ON citation_edges(cited_paper_id)
        """)
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        return False

def drop_indexes(db_path: str) -> bool:
    """
    Drop performance indexes from citation tables.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if indexes were dropped successfully
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of indexes (excluding SQLite internal ones)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Drop each index
        for index_name in indexes:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error dropping indexes: {e}")
        return False

def backup_database(source_path: str, backup_path: str) -> bool:
    """
    Create a backup of the database.
    
    Args:
        source_path: Path to source database
        backup_path: Path for backup database
        
    Returns:
        bool: True if backup succeeded, False otherwise
    """
    try:
        import shutil
        shutil.copy2(source_path, backup_path)
        return True
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return False

def restore_database(backup_path: str, target_path: str) -> bool:
    """
    Restore database from backup.
    
    Args:
        backup_path: Path to backup database
        target_path: Path for restored database
        
    Returns:
        bool: True if restore succeeded, False otherwise
    """
    try:
        import shutil
        shutil.copy2(backup_path, target_path)
        return True
    except Exception as e:
        logger.error(f"Error restoring database: {e}")
        return False

# Main execution for testing
if __name__ == "__main__":
    # Test schema creation
    test_db = "test_citations.db"
    
    print("Creating citation tracking schema...")
    success = create_citation_tables(test_db)
    
    if success:
        print("Schema created successfully!")
        
        # Verify schema
        print("\nVerifying schema...")
        verification = verify_schema(test_db)
        print(json.dumps(verification, indent=2))
        
        # Get statistics
        print("\nDatabase statistics...")
        stats = get_database_statistics(test_db)
        print(json.dumps(stats, indent=2))
        
        # Export DDL
        print("\nExporting schema DDL...")
        export_schema_ddl(test_db, "test_schema.sql")
        
    else:
        print("Failed to create schema!")
