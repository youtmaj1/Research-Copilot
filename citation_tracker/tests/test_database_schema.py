"""
Tests for Database Schema Module

This module tests the database schema functionality including:
- Schema creation and validation
- Table structure verification
- Index creation and management
- Database migrations
- Schema backup and restore
- Performance optimization
"""

import pytest
import tempfile
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from database_schema import (
    create_citation_tables, verify_schema, backup_database, 
    restore_database, optimize_database, migrate_schema,
    get_table_info, create_indexes, drop_indexes
)

class TestSchemaCreation:
    """Test database schema creation functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary database."""
        self.temp_db = tempfile.mktemp(suffix='.db')
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_create_citation_tables_basic(self):
        """Test basic citation tables creation."""
        success = create_citation_tables(self.temp_db)
        assert success, "Schema creation should succeed"
        
        # Verify database file exists
        assert os.path.exists(self.temp_db)
        
        # Verify tables were created
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = [
            'citation_bursts',
            'citation_edges',
            'citation_history',
            'citation_matches',
            'citation_nodes',
            'citation_processing_log',
            'citation_schema_info',
            'extracted_citations',
            'graph_snapshots',
            'trending_papers'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} should be created"
    
    def test_create_citation_tables_idempotent(self):
        """Test that creating tables multiple times is safe."""
        # Create tables first time
        success1 = create_citation_tables(self.temp_db)
        assert success1
        
        # Create tables second time (should not fail)
        success2 = create_citation_tables(self.temp_db)
        assert success2
        
        # Verify tables still exist and work
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Insert test data
        cursor.execute("""
            INSERT INTO extracted_citations (source_paper_id, raw_text)
            VALUES (?, ?)
        """, ('test_paper', 'test citation'))
        
        # Verify insertion worked
        cursor.execute("SELECT COUNT(*) FROM extracted_citations")
        count = cursor.fetchone()[0]
        assert count == 1
        
        conn.commit()
        conn.close()
    
    def test_table_structures(self):
        """Test that tables have correct structures."""
        create_citation_tables(self.temp_db)
        
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Test extracted_citations table structure
        cursor.execute("PRAGMA table_info(extracted_citations)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}  # name: type
        
        expected_columns = {
            'id': 'INTEGER',
            'source_paper_id': 'TEXT',
            'raw_text': 'TEXT',
            'doi': 'TEXT',
            'arxiv_id': 'TEXT',
            'title': 'TEXT',
            'authors': 'TEXT',
            'year': 'INTEGER',
            'venue': 'TEXT',
            'confidence': 'REAL',
            'extracted_at': 'TEXT'
        }
        
        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column {col_name} should exist"
            # SQLite type matching can be flexible, so just check main types
            assert col_type.upper() in columns[col_name].upper() or columns[col_name].upper() in col_type.upper()
        
        conn.close()
    
    def test_foreign_key_constraints(self):
        """Test foreign key constraints are properly set up."""
        create_citation_tables(self.temp_db)
        
        conn = sqlite3.connect(self.temp_db)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key checks
        cursor = conn.cursor()
        
        # Insert test data to verify foreign key relationships
        # First insert into extracted_citations
        cursor.execute("""
            INSERT INTO extracted_citations (id, source_paper_id, raw_text)
            VALUES (1, 'paper1', 'test citation')
        """)
        
        # Now insert into citation_matches referencing the citation
        cursor.execute("""
            INSERT INTO citation_matches (citation_id, paper_id, match_type, confidence)
            VALUES (1, 'matched_paper', 'title', 0.8)
        """)
        
        # This should work without foreign key violations
        conn.commit()
        
        # Try to insert with invalid foreign key - should fail
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO citation_matches (citation_id, paper_id, match_type, confidence)
                VALUES (999, 'nonexistent_paper', 'title', 0.8)
            """)
            conn.commit()
        
        conn.close()

class TestSchemaVerification:
    """Test schema verification functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_verify_complete_schema(self):
        """Test verifying a complete, correct schema."""
        create_citation_tables(self.temp_db)
        
        schema_info = verify_schema(self.temp_db)
        
        assert schema_info['schema_exists'] == True
        assert len(schema_info['missing_tables']) == 0
        assert len(schema_info['missing_indexes']) == 0
        assert schema_info['foreign_keys_enabled'] == True
    
    def test_verify_missing_tables(self):
        """Test verification with missing tables."""
        # Create database with only some tables
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Create only one table
        cursor.execute("""
            CREATE TABLE extracted_citations (
                id INTEGER PRIMARY KEY,
                source_paper_id TEXT,
                raw_text TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        schema_info = verify_schema(self.temp_db)
        
        assert schema_info['schema_exists'] == False
        assert len(schema_info['missing_tables']) > 0
        assert 'citation_matches' in schema_info['missing_tables']
    
    def test_verify_nonexistent_database(self):
        """Test verification with non-existent database."""
        schema_info = verify_schema('nonexistent.db')
        
        assert schema_info['schema_exists'] == False
        assert schema_info['error'] is not None
    
    def test_get_table_info(self):
        """Test getting table information."""
        create_citation_tables(self.temp_db)
        
        table_info = get_table_info(self.temp_db, 'extracted_citations')
        
        assert table_info is not None
        assert 'columns' in table_info
        assert 'indexes' in table_info
        assert len(table_info['columns']) > 0
        
        # Check for expected columns
        column_names = [col['name'] for col in table_info['columns']]
        assert 'id' in column_names
        assert 'source_paper_id' in column_names
        assert 'raw_text' in column_names

class TestIndexManagement:
    """Test database index management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        create_citation_tables(self.temp_db)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_create_indexes(self):
        """Test creating database indexes."""
        success = create_indexes(self.temp_db)
        assert success
        
        # Verify indexes were created
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Should have several indexes
        assert len(indexes) > 0
        
        # Check for specific important indexes
        expected_patterns = ['citations', 'snapshots', 'matches']
        found_patterns = sum(1 for pattern in expected_patterns 
                           if any(pattern in idx for idx in indexes))
        assert found_patterns > 0, "Should have indexes for key tables"
    
    def test_drop_indexes(self):
        """Test dropping database indexes."""
        # First create indexes
        create_indexes(self.temp_db)
        
        # Verify they exist
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        index_count_before = cursor.fetchone()[0]
        conn.close()
        
        assert index_count_before > 0
        
        # Drop indexes
        success = drop_indexes(self.temp_db)
        assert success
        
        # Verify they're gone
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        index_count_after = cursor.fetchone()[0]
        conn.close()
        
        assert index_count_after < index_count_before

class TestDatabaseOperations:
    """Test database backup, restore, and optimization."""
    
    def setup_method(self):
        """Set up test environment with sample data."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.backup_db = tempfile.mktemp(suffix='_backup.db')
        
        create_citation_tables(self.temp_db)
        self.populate_test_data()
    
    def teardown_method(self):
        """Clean up test environment."""
        for db_path in [self.temp_db, self.backup_db]:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def populate_test_data(self):
        """Populate database with test data."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Insert sample citations
        test_citations = [
            ('paper1', 'Smith, J. et al. Research Paper. Journal 2020.'),
            ('paper2', 'Jones, A. Another Study. Conference 2021.'),
            ('paper3', 'Brown, B. Advanced Methods. Journal 2022.'),
        ]
        
        cursor.executemany("""
            INSERT INTO extracted_citations (source_paper_id, raw_text)
            VALUES (?, ?)
        """, test_citations)
        
        # Insert sample snapshots
        base_time = datetime.now() - timedelta(days=30)
        for i, paper_id in enumerate(['paper1', 'paper2', 'paper3']):
            for day in range(0, 30, 5):
                timestamp = base_time + timedelta(days=day)
                citation_count = 10 + i * 5 + day
                cursor.execute("""
                    INSERT INTO citation_snapshots (paper_id, citation_count, timestamp)
                    VALUES (?, ?, ?)
                """, (paper_id, citation_count, timestamp.isoformat()))
        
        conn.commit()
        conn.close()
    
    def test_backup_database(self):
        """Test database backup functionality."""
        success = backup_database(self.temp_db, self.backup_db)
        assert success, "Backup should succeed"
        
        # Verify backup file exists
        assert os.path.exists(self.backup_db)
        
        # Verify backup has same structure
        schema_original = verify_schema(self.temp_db)
        schema_backup = verify_schema(self.backup_db)
        
        assert schema_original['schema_exists'] == schema_backup['schema_exists']
        
        # Verify data was copied
        conn_orig = sqlite3.connect(self.temp_db)
        conn_backup = sqlite3.connect(self.backup_db)
        
        cursor_orig = conn_orig.cursor()
        cursor_backup = conn_backup.cursor()
        
        cursor_orig.execute("SELECT COUNT(*) FROM extracted_citations")
        count_orig = cursor_orig.fetchone()[0]
        
        cursor_backup.execute("SELECT COUNT(*) FROM extracted_citations")
        count_backup = cursor_backup.fetchone()[0]
        
        assert count_orig == count_backup, "Backup should have same data count"
        
        conn_orig.close()
        conn_backup.close()
    
    def test_restore_database(self):
        """Test database restore functionality."""
        # First create backup
        backup_database(self.temp_db, self.backup_db)
        
        # Modify original database
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM extracted_citations")
        conn.commit()
        conn.close()
        
        # Verify original is modified
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM extracted_citations")
        count_after_delete = cursor.fetchone()[0]
        conn.close()
        
        assert count_after_delete == 0
        
        # Restore from backup
        success = restore_database(self.backup_db, self.temp_db)
        assert success, "Restore should succeed"
        
        # Verify data is restored
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM extracted_citations")
        count_after_restore = cursor.fetchone()[0]
        conn.close()
        
        assert count_after_restore > 0, "Data should be restored"
    
    def test_optimize_database(self):
        """Test database optimization."""
        # Get database size before optimization
        size_before = os.path.getsize(self.temp_db)
        
        success = optimize_database(self.temp_db)
        assert success, "Optimization should succeed"
        
        # Database should still exist and be functional
        assert os.path.exists(self.temp_db)
        
        # Verify data is still there
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM extracted_citations")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0, "Data should still exist after optimization"

class TestSchemaMigration:
    """Test database schema migration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_migrate_schema_new_database(self):
        """Test migrating schema on new database."""
        success = migrate_schema(self.temp_db, target_version=1)
        assert success, "Migration should succeed on new database"
        
        # Verify tables were created
        schema_info = verify_schema(self.temp_db)
        assert schema_info['schema_exists'] == True
    
    def test_migrate_schema_existing_database(self):
        """Test migrating schema on existing database."""
        # Create initial schema
        create_citation_tables(self.temp_db)
        
        # Add some data
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO extracted_citations (source_paper_id, raw_text)
            VALUES ('test', 'test citation')
        """)
        conn.commit()
        conn.close()
        
        # Migrate schema (should be idempotent)
        success = migrate_schema(self.temp_db, target_version=1)
        assert success
        
        # Verify data is preserved
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM extracted_citations")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1, "Data should be preserved during migration"

class TestSchemaPerformance:
    """Test schema performance with larger datasets."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        create_citation_tables(self.temp_db)
        create_indexes(self.temp_db)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_bulk_insert_performance(self):
        """Test performance of bulk inserts."""
        import time
        
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Prepare large dataset
        test_data = []
        for i in range(1000):
            test_data.append((
                f'paper_{i}',
                f'Citation text {i} with some content to make it realistic',
                f'10.1000/{i}',
                f'Test Title {i}',
                0.8 + (i % 20) * 0.01
            ))
        
        # Time the bulk insert
        start_time = time.time()
        cursor.executemany("""
            INSERT INTO extracted_citations 
            (source_paper_id, raw_text, doi, title, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, test_data)
        conn.commit()
        insert_time = time.time() - start_time
        
        conn.close()
        
        assert insert_time < 5, f"Bulk insert took too long: {insert_time}s"
    
    def test_query_performance_with_indexes(self):
        """Test query performance with indexes."""
        import time
        
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Insert test data
        for i in range(500):
            cursor.execute("""
                INSERT INTO extracted_citations (source_paper_id, raw_text, doi)
                VALUES (?, ?, ?)
            """, (f'paper_{i % 50}', f'Citation {i}', f'10.1000/{i}'))
        
        conn.commit()
        
        # Test query performance
        start_time = time.time()
        cursor.execute("""
            SELECT * FROM extracted_citations 
            WHERE source_paper_id = ? 
            ORDER BY created_at DESC
        """, ('paper_1',))
        results = cursor.fetchall()
        query_time = time.time() - start_time
        
        conn.close()
        
        assert query_time < 1, f"Query took too long: {query_time}s"
        assert len(results) > 0, "Should find matching records"
    
    def test_complex_join_performance(self):
        """Test performance of complex joins."""
        import time
        
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Insert related data
        for i in range(100):
            # Insert citation
            cursor.execute("""
                INSERT INTO extracted_citations (id, source_paper_id, raw_text)
                VALUES (?, ?, ?)
            """, (i, f'paper_{i}', f'Citation {i}'))
            
            # Insert match
            cursor.execute("""
                INSERT INTO citation_matches (citation_id, paper_id, match_type, confidence)
                VALUES (?, ?, ?, ?)
            """, (i, f'matched_paper_{i}', 'title', 0.8))
        
        conn.commit()
        
        # Test complex join query
        start_time = time.time()
        cursor.execute("""
            SELECT c.source_paper_id, c.raw_text, m.paper_id, m.confidence
            FROM extracted_citations c
            JOIN citation_matches m ON c.id = m.citation_id
            WHERE m.confidence > 0.7
            ORDER BY m.confidence DESC
            LIMIT 50
        """)
        results = cursor.fetchall()
        query_time = time.time() - start_time
        
        conn.close()
        
        assert query_time < 2, f"Complex join took too long: {query_time}s"
        assert len(results) > 0, "Should find matching records"

class TestSchemaErrorHandling:
    """Test error handling in schema operations."""
    
    def test_create_tables_on_readonly_database(self):
        """Test creating tables on read-only database."""
        # This is difficult to test portably, but we can test error handling
        success = create_citation_tables('')  # Invalid path
        assert success == False, "Should fail gracefully with invalid path"
    
    def test_verify_schema_corrupted_database(self):
        """Test schema verification on corrupted database."""
        # Create a file that's not a valid SQLite database
        fake_db = tempfile.mktemp(suffix='.db')
        
        try:
            with open(fake_db, 'w') as f:
                f.write("This is not a database file")
            
            schema_info = verify_schema(fake_db)
            assert schema_info['schema_exists'] == False
            assert schema_info['error'] is not None
            
        finally:
            if os.path.exists(fake_db):
                os.unlink(fake_db)
    
    def test_backup_nonexistent_database(self):
        """Test backing up non-existent database."""
        success = backup_database('nonexistent.db', 'backup.db')
        assert success == False, "Should fail gracefully with non-existent source"
    
    def test_optimize_invalid_database(self):
        """Test optimizing invalid database."""
        success = optimize_database('nonexistent.db')
        assert success == False, "Should fail gracefully with non-existent database"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
