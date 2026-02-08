"""
SQLite Database Manager
=======================
Manages all database operations with full schema and CRUD operations.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager with full schema support.
    
    Tables:
    - conversations: Customer support conversations
    - tickets: Support tickets
    - scripts: Support scripts/runbooks  
    - knowledge_articles: Generated KB articles
    - kb_versions: Article version history
    - kb_lineage: Article provenance
    - questions: Ground truth Q&A pairs
    - placeholders: Template placeholders
    - qa_scores: Quality assessment scores
    - learning_events: Learning system events
    - evaluation_runs: Evaluation records
    - learning_checkpoints: Training checkpoints
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config['database']['path']
        self.backup_enabled = config['database'].get('backup_enabled', True)
        self.backup_dir = config['database'].get('backup_dir', 'data/backups')
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def initialize(self) -> None:
        """Initialize database with full schema."""
        logger.info("Initializing database schema...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_number TEXT,
                    conversation_id TEXT UNIQUE,
                    channel TEXT,
                    conversation_start TIMESTAMP,
                    conversation_end TIMESTAMP,
                    customer_role TEXT,
                    agent_name TEXT,
                    product TEXT,
                    category TEXT,
                    issue_summary TEXT,
                    transcript TEXT,
                    sentiment TEXT,
                    generation_source_record TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tickets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_number TEXT UNIQUE,
                    status TEXT,
                    priority TEXT,
                    tier TEXT,
                    created_date TIMESTAMP,
                    resolved_date TIMESTAMP,
                    resolution_summary TEXT,
                    contact_email TEXT,
                    contact_phone TEXT,
                    contact_name TEXT,
                    contact_role TEXT,
                    account_name TEXT,
                    property_name TEXT,
                    product TEXT,
                    module TEXT,
                    category TEXT,
                    case_type TEXT,
                    subject TEXT,
                    description TEXT,
                    root_cause TEXT,
                    tags TEXT,
                    kb_article_id TEXT,
                    script_id TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Scripts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    script_id TEXT UNIQUE NOT NULL,
                    script_name TEXT NOT NULL,
                    script_content TEXT NOT NULL,
                    product TEXT,
                    category TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Knowledge Articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_article_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    content TEXT NOT NULL,
                    product TEXT,
                    category TEXT,
                    author TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'draft',
                    embedding_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # KB Versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kb_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_article_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    change_summary TEXT,
                    confidence REAL,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kb_article_id) REFERENCES knowledge_articles(kb_article_id),
                    UNIQUE(kb_article_id, version)
                )
            ''')
            
            # KB Lineage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kb_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lineage_id TEXT UNIQUE NOT NULL,
                    kb_article_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    version INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kb_article_id) REFERENCES knowledge_articles(kb_article_id)
                )
            ''')
            
            # Questions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id TEXT UNIQUE,
                    question_text TEXT,
                    answer_type TEXT,
                    target_id TEXT,
                    target_title TEXT,
                    source TEXT,
                    product TEXT,
                    category TEXT,
                    module TEXT,
                    difficulty TEXT,
                    generation_source_record TEXT,
                    answer_text TEXT,
                    confidence REAL DEFAULT 1.0,
                    version INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Placeholders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS placeholders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    placeholder_id TEXT UNIQUE NOT NULL,
                    placeholder_name TEXT NOT NULL,
                    description TEXT,
                    default_value TEXT,
                    script_id TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (script_id) REFERENCES scripts(script_id)
                )
            ''')
            
            # QA Scores table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_article_id TEXT NOT NULL,
                    evaluation_run_id TEXT,
                    accuracy_score REAL,
                    completeness_score REAL,
                    clarity_score REAL,
                    compliance_score REAL,
                    overall_score REAL,
                    violations TEXT,
                    feedback TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kb_article_id) REFERENCES knowledge_articles(kb_article_id)
                )
            ''')
            
            # Learning Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    kb_article_id TEXT,
                    details TEXT,
                    status TEXT DEFAULT 'pending',
                    confidence REAL,
                    approved_by TEXT,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kb_article_id) REFERENCES knowledge_articles(kb_article_id)
                )
            ''')
            
            # Evaluation Runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    run_type TEXT NOT NULL,
                    metrics TEXT,
                    samples_evaluated INTEGER,
                    duration_seconds REAL,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Learning Checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_id TEXT UNIQUE NOT NULL,
                    batch_id TEXT,
                    round_number INTEGER,
                    model_version TEXT,
                    prompt_version TEXT,
                    metrics TEXT,
                    state_path TEXT,
                    version INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_ticket ON conversations(ticket_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kb_status ON knowledge_articles(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kb_product ON knowledge_articles(product)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(answer_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_kb ON kb_lineage(kb_article_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qa_scores_kb ON qa_scores(kb_article_id)')
            
            conn.commit()
            
        logger.info("Database schema initialized successfully")
    
    def insert_dataframe(self, table: str, df, conflict_action: str = 'IGNORE') -> int:
        """
        Insert a pandas DataFrame into a table.
        
        Args:
            table: Table name
            df: DataFrame to insert
            conflict_action: IGNORE, REPLACE, or ABORT
            
        Returns:
            Number of rows inserted
        """
        import pandas as pd
        
        if df.empty:
            return 0
        
        # Convert column names to snake_case
        df = df.copy()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Column name mappings from Excel to database
        column_mappings = {
            # Conversations sheet mappings
            'transcript': 'transcript',
            'conversation_start': 'conversation_start',
            'conversation_end': 'conversation_end',
            # Tickets sheet mappings
            'created_at': 'created_date',
            'closed_at': 'resolved_date',
            'subject': 'issue_summary',
            'description': 'resolution_summary',
            'resolution': 'resolution_summary',
            # Scripts sheet mappings  
            'script_title': 'script_name',
            'script_text_sanitized': 'script_content',
            'script_purpose': 'description',
            # Knowledge Articles mappings
            'body': 'content',
            'article_body': 'content',
            # Questions sheet mappings
            'question_text': 'question_text',
            'answer_type': 'answer_type',
            'target_id': 'target_id',
            'target_title': 'answer_text',
            # Placeholder mappings
            'placeholder': 'placeholder_name',
            'meaning': 'description',
            'example': 'default_value',
            # Learning Events mappings
            'detected_gap': 'details',
            'final_status': 'status',
            'proposed_kb_article_id': 'kb_article_id',
            'draft_summary': 'event_type',
            'event_timestamp': 'created_date',
            'reviewer_role': 'approved_by',
        }
        
        # Apply column mappings
        df = df.rename(columns=column_mappings)
        
        # Auto-generate missing required IDs
        if table == 'kb_lineage' and 'lineage_id' not in df.columns:
            df['lineage_id'] = [f"LIN-{i:06d}" for i in range(len(df))]
        if table == 'placeholders':
            if 'placeholder_id' not in df.columns:
                df['placeholder_id'] = [f"PH-{i:04d}" for i in range(len(df))]
            if 'placeholder_name' not in df.columns and 'placeholder' in df.columns:
                df['placeholder_name'] = df['placeholder']
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        with self.get_connection() as conn:
            # Get existing columns
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            # Filter to matching columns (ensure unique)
            matching_cols = list(dict.fromkeys([col for col in df.columns if col in existing_cols]))
            df_filtered = df[matching_cols]
            
            # Build insert query
            cols = ', '.join(matching_cols)
            placeholders = ', '.join(['?' for _ in matching_cols])
            query = f"INSERT OR {conflict_action} INTO {table} ({cols}) VALUES ({placeholders})"
            
            # Insert rows
            rows_before = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            
            for _, row in df_filtered.iterrows():
                values = []
                for col in matching_cols:
                    val = row[col]
                    
                    # If val is a Series (duplicate cols), take first value
                    if isinstance(val, pd.Series):
                        val = val.iloc[0] if len(val) > 0 else None
                    
                    # Check for null values
                    if val is None:
                        values.append(None)
                    elif isinstance(val, float) and pd.isna(val):
                        values.append(None)
                    elif hasattr(val, 'isoformat'):  # Handle Timestamp/datetime
                        values.append(val.isoformat())
                    elif isinstance(val, (list, dict)):
                        values.append(str(val))
                    else:
                        values.append(val)
                try:
                    cursor.execute(query, values)
                except sqlite3.IntegrityError as e:
                    if conflict_action == 'ABORT':
                        raise
                    logger.debug(f"Skipping duplicate row: {e}")
            
            rows_after = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            
        return rows_after - rows_before
    
    def get_all(self, table: str, where: Optional[str] = None, 
                params: Optional[Tuple] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get all rows from a table."""
        query = f"SELECT * FROM {table}"
        
        if where:
            query += f" WHERE {where}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_by_id(self, table: str, id_column: str, id_value: str) -> Optional[Dict]:
        """Get a single row by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} WHERE {id_column} = ?", (id_value,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update(self, table: str, id_column: str, id_value: str, updates: Dict[str, Any]) -> bool:
        """Update a row."""
        if not updates:
            return False
        
        updates['updated_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [id_value]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {table} SET {set_clause} WHERE {id_column} = ?", values)
            return cursor.rowcount > 0
    
    def delete(self, table: str, id_column: str, id_value: str) -> bool:
        """Delete a row."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (id_value,))
            return cursor.rowcount > 0
    
    def count(self, table: str, where: Optional[str] = None, params: Optional[Tuple] = None) -> int:
        """Count rows in a table."""
        query = f"SELECT COUNT(*) FROM {table}"
        if where:
            query += f" WHERE {where}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor.fetchone()[0]
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """Execute a custom query."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            
            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            return []
    
    def backup(self) -> str:
        """Create a database backup."""
        if not self.backup_enabled:
            return ""
        
        backup_path = Path(self.backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_path / f"support_backup_{timestamp}.db"
        
        with self.get_connection() as conn:
            backup_conn = sqlite3.connect(backup_file)
            conn.backup(backup_conn)
            backup_conn.close()
        
        logger.info(f"Database backed up to: {backup_file}")
        return str(backup_file)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        tables = [
            'conversations', 'tickets', 'scripts', 'knowledge_articles',
            'kb_versions', 'kb_lineage', 'questions', 'placeholders',
            'qa_scores', 'learning_events', 'evaluation_runs', 'learning_checkpoints'
        ]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    stats[table] = 0
        
        stats['total_records'] = sum(stats.values())
        
        return stats
