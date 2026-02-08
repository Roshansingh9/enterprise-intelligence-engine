"""
Excel Ingestion Module
======================
Parses multi-sheet Excel files and extracts structured data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ExcelIngestion:
    """
    Ingests Excel files with multiple sheets and preserves relationships.
    
    Handles:
    - Conversations ↔ Tickets (join on Ticket_Number)
    - Questions → Answer sources (SCRIPT, KB, TICKET_RESOLUTION)
    - KB Lineage tracking
    - Learning Events
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config['ingestion'].get('batch_size', 100)
        self.validate_schema = config['ingestion'].get('validate_schema', True)
        self.skip_invalid = config['ingestion'].get('skip_invalid_rows', False)
        
        # Expected sheets and their schemas
        self.sheet_schemas = {
            'Conversations': {
                'required': ['Ticket_Number', 'Conversation_ID', 'Channel', 'Transcript'],
                'optional': ['Conversation_Start', 'Conversation_End', 'Customer_Role', 
                           'Agent_Name', 'Product', 'Category', 'Issue_Summary', 'Sentiment']
            },
            'Tickets': {
                'required': ['Ticket_Number', 'Status'],
                'optional': ['Priority', 'Created_Date', 'Resolved_Date', 'Resolution_Summary',
                           'Contact_Email', 'Contact_Phone', 'Product', 'Category']
            },
            'Scripts_Master': {
                'required': ['Script_ID', 'Script_Name', 'Script_Content'],
                'optional': ['Product', 'Category', 'Version', 'Status', 'Created_Date']
            },
            'Knowledge_Articles': {
                'required': ['KB_Article_ID', 'Title', 'Content'],
                'optional': ['Summary', 'Product', 'Category', 'Version', 'Status',
                           'Created_Date', 'Updated_Date', 'Author']
            },
            'Questions': {
                'required': ['Question_ID', 'Question_Text', 'Answer_Type', 'Target_ID'],
                'optional': ['Answer_Text', 'Confidence', 'Created_Date']
            },
            'KB_Lineage': {
                'required': ['Lineage_ID', 'KB_Article_ID', 'Source_Type', 'Source_ID'],
                'optional': ['Confidence', 'Created_Date']
            },
            'Learning_Events': {
                'required': ['Event_ID', 'Event_Type', 'KB_Article_ID'],
                'optional': ['Status', 'Confidence', 'Created_Date', 'Approved_By']
            },
            'Placeholders': {
                'required': ['Placeholder_ID', 'Placeholder_Name', 'Description'],
                'optional': ['Default_Value', 'Script_ID']
            }
        }
    
    def ingest(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """
        Ingest all sheets from an Excel file.
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            Dictionary mapping sheet names to DataFrames
        """
        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        logger.info(f"Ingesting Excel file: {excel_path}")
        
        # Read all sheets
        excel_file = pd.ExcelFile(path)
        sheets = {}
        
        for sheet_name in excel_file.sheet_names:
            logger.info(f"Reading sheet: {sheet_name}")
            
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Skip info/readme sheets
                if df.empty or len(df.columns) < 2:
                    logger.warning(f"Skipping empty or info sheet: {sheet_name}")
                    continue
                
                # Clean column names
                df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
                
                # Validate schema if enabled
                if self.validate_schema and sheet_name in self.sheet_schemas:
                    df = self._validate_and_clean(df, sheet_name)
                
                # Add metadata columns
                df = self._add_metadata(df)
                
                sheets[sheet_name] = df
                logger.info(f"  -> {len(df)} rows loaded")
                
            except Exception as e:
                logger.error(f"Error reading sheet {sheet_name}: {e}")
                if not self.skip_invalid:
                    raise
        
        # Validate relationships
        self._validate_relationships(sheets)
        
        return sheets
    
    def _validate_and_clean(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Validate DataFrame against expected schema."""
        schema = self.sheet_schemas[sheet_name]
        
        # Check required columns
        missing = set(schema['required']) - set(df.columns)
        if missing:
            logger.warning(f"Missing required columns in {sheet_name}: {missing}")
            # Add missing columns with None
            for col in missing:
                df[col] = None
        
        # Clean data
        df = df.dropna(how='all')  # Remove completely empty rows
        
        # Ensure ID columns are strings
        id_columns = [col for col in df.columns if col.endswith('_ID') or col.endswith('_Number')]
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard metadata columns."""
        now = datetime.now().isoformat()
        
        if 'ingested_at' not in df.columns:
            df['ingested_at'] = now
        
        if 'version' not in df.columns:
            df['version'] = 1
        
        if 'status' not in df.columns:
            df['status'] = 'active'
        
        return df
    
    def _validate_relationships(self, sheets: Dict[str, pd.DataFrame]) -> None:
        """Validate foreign key relationships between sheets."""
        
        # Conversations ↔ Tickets
        if 'Conversations' in sheets and 'Tickets' in sheets:
            conv_tickets = set(sheets['Conversations']['Ticket_Number'].dropna().unique())
            ticket_ids = set(sheets['Tickets']['Ticket_Number'].dropna().unique())
            orphan_convs = conv_tickets - ticket_ids
            if orphan_convs:
                logger.warning(f"Conversations with missing tickets: {len(orphan_convs)}")
        
        # Questions → Target sources
        if 'Questions' in sheets:
            questions = sheets['Questions']
            for answer_type in questions['Answer_Type'].unique():
                if answer_type == 'SCRIPT' and 'Scripts_Master' in sheets:
                    target_ids = set(questions[questions['Answer_Type'] == 'SCRIPT']['Target_ID'])
                    script_ids = set(sheets['Scripts_Master']['Script_ID'])
                    missing = target_ids - script_ids
                    if missing:
                        logger.warning(f"Questions referencing missing scripts: {len(missing)}")
        
        logger.info("Relationship validation complete")
    
    def get_joined_data(self, sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a joined view of conversations with ticket context.
        
        Returns:
            DataFrame with full conversation context
        """
        if 'Conversations' not in sheets:
            raise ValueError("Conversations sheet required for join")
        
        result = sheets['Conversations'].copy()
        
        # Join with Tickets
        if 'Tickets' in sheets:
            tickets = sheets['Tickets'][['Ticket_Number', 'Status', 'Priority', 
                                         'Resolution_Summary', 'Created_Date', 'Resolved_Date']]
            result = result.merge(tickets, on='Ticket_Number', how='left', suffixes=('', '_ticket'))
        
        return result


class DataValidator:
    """Validates and cleans ingested data."""
    
    @staticmethod
    def validate_transcript(transcript: str) -> Tuple[bool, Optional[str]]:
        """Validate a conversation transcript."""
        if not transcript or not isinstance(transcript, str):
            return False, "Empty or invalid transcript"
        
        if len(transcript) < 50:
            return False, "Transcript too short"
        
        return True, None
    
    @staticmethod
    def validate_kb_article(article: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a KB article structure."""
        errors = []
        
        if not article.get('Title'):
            errors.append("Missing title")
        
        if not article.get('Content'):
            errors.append("Missing content")
        elif len(article['Content']) < 100:
            errors.append("Content too short")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excess whitespace
        text = ' '.join(text.split())
        
        # Remove common artifacts
        text = text.replace('<br/>', '\n')
        text = text.replace('<br>', '\n')
        
        return text.strip()
