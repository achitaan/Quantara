import chainlit as cl
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Optional
import re
import time
import json
import hashlib
import pickle
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# PostgreSQL imports
import psycopg2
from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager

load_dotenv()  # ← makes OPENAI_API_KEY available for both LLM & embeddings

# ── Configuration Classes ─────────────────────────────────────────────────
@dataclass
class LLMConfig:
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    streaming: bool = True
    timeout: int = 30

@dataclass
class ReflectionConfig:
    enabled: bool = True
    min_score_threshold: float = 7.0
    max_iterations: int = 2
    use_cheaper_model: bool = True

@dataclass
class RetrievalConfig:
    mode: str = "hybrid"  # basic, hybrid, rerank, compressed
    k: int = 6
    enable_compression: bool = False
    rerank_top_k: int = 3

@dataclass
class UIConfig:
    thinking_display_time: float = 2.0  # seconds to display thinking before answer
    auto_open_reflection: bool = True
    show_thinking_before_answer: bool = True

@dataclass
class CacheConfig:
    enable_response_cache: bool = True
    enable_embedding_cache: bool = True
    enable_retrieval_cache: bool = True
    enable_thinking_cache: bool = False  # Optional for thinking processes
    cache_ttl_hours: int = 24  # Time to live
    max_cache_size_mb: int = 100  # Maximum cache size
    
    # Database configuration
    use_postgresql: bool = True  # Use PostgreSQL instead of SQLite
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "quantara"
    postgres_user: str = "root"
    postgres_password: str = "1412"
    postgres_schema: str = "cache"
    
    # Connection pool settings
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_pool_timeout: int = 30
    
    # Fallback SQLite (if PostgreSQL unavailable)
    cache_db_path: str = "cache/quantara_cache.db"

@dataclass
class QuantaraConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    debug_mode: bool = False

# Global configuration
config = QuantaraConfig()

# Thread-safe chain storage to avoid serialization issues
import threading
chain_storage = threading.local()

# ── LangChain / LangGraph imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# ── Tool layer imports ───────────────────────────────────────────────────────
from tools import (
    get_tool_definitions, 
    get_tool_handler, 
    get_tool_info,
    TOOL_HANDLERS
)
import json

# ── Utility Functions ────────────────────────────────────────────────────────
import time
import asyncio
from functools import wraps

def monitor_performance(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            if config.debug_mode:
                print(f"{operation_name} took {end_time - start_time:.2f} seconds")
            
            return result
        return wrapper
    return decorator

# ── Structured Output Models ────────────────────────────────────────────────
class RegulatoryAnalysis(BaseModel):
    """Structured output for regulatory analysis."""
    key_requirements: List[str] = Field(..., description="Main regulatory requirements")
    implementation_steps: List[str] = Field(..., description="Steps for implementation")
    risk_factors: List[str] = Field(..., description="Key risk considerations")
    best_practices: List[str] = Field(..., description="Industry best practices")
    recent_updates: Optional[List[str]] = Field(None, description="Recent regulatory changes")

class FinancialAnalysis(BaseModel):
    """Structured output for financial analysis."""
    executive_summary: str = Field(..., description="Brief executive summary")
    key_metrics: List[str] = Field(..., description="Important financial metrics")
    analysis_points: List[str] = Field(..., description="Main analysis findings")
    risks: List[str] = Field(..., description="Identified risks")
    recommendations: List[str] = Field(..., description="Actionable recommendations")

class PortfolioAnalysis(BaseModel):
    """Structured output for portfolio analysis."""
    portfolio_type: str = Field(..., description="Type of portfolio")
    allocation_breakdown: List[str] = Field(..., description="Asset allocation details")
    risk_metrics: List[str] = Field(..., description="Risk assessment metrics")
    performance_indicators: List[str] = Field(..., description="Performance metrics")
    optimization_suggestions: List[str] = Field(..., description="Improvement suggestions")

# ── LLM Performance Monitor ──────────────────────────────────────────────────
class LLMPerformanceMonitor:
    """Monitor LLM performance and costs."""
    def __init__(self):
        self.requests: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.response_times: list = []
        self.model_usage: dict = {}
        self.error_count: int = 0
        
    def log_request(self, model: str, prompt_tokens: int, completion_tokens: int, 
                   response_time: float, error: bool = False, cache_hit: bool = False):
        """Log a single LLM request"""
        if error:
            self.error_count += 1
            return
            
        self.requests += 1
        tokens = prompt_tokens + completion_tokens
        self.total_tokens += tokens
        
        # Only add to cost if not a cache hit
        if not cache_hit:
            # Calculate approximate cost
            if "gpt-4" in model and "mini" not in model:
                prompt_cost = prompt_tokens * 0.00001  # $0.01 per 1K tokens
                completion_cost = completion_tokens * 0.00003  # $0.03 per 1K tokens
            else:  # gpt-4o-mini or similar
                prompt_cost = prompt_tokens * 0.000005  # $0.005 per 1K tokens
                completion_cost = completion_tokens * 0.000015  # $0.015 per 1K tokens
            
            request_cost = prompt_cost + completion_cost
            self.total_cost += request_cost
        
        # Store response time
        self.response_times.append(response_time)
        
        # Track model usage
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
        
    def get_statistics(self):
        """Get performance statistics"""
        if not self.response_times:
            return {"error": "No data collected"}
            
        avg_response_time = sum(self.response_times) / len(self.response_times)
        return {
            "total_requests": self.requests,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "avg_response_time_sec": round(avg_response_time, 2),
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(self.requests + self.error_count, 1), 3),
            "model_usage": self.model_usage,
            "avg_tokens_per_request": round(self.total_tokens / max(self.requests, 1), 1)
        }

# Create singleton instance
performance_monitor = LLMPerformanceMonitor()

# ── Advanced Cache Manager with PostgreSQL ──────────────────────────────────
class AdvancedCacheManager:
    """Multi-layer caching system with PostgreSQL backend and memory cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Memory cache (LRU-style with size limits)
        self.memory_cache = {}
        self.cache_access_times = {}
        self.cache_sizes = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0, 'misses': 0, 'memory_hits': 0, 'db_hits': 0,
            'total_cached_items': 0, 'memory_usage_mb': 0
        }
        
        # Initialize database connection
        if config.use_postgresql:
            try:
                self._init_postgresql()
                self.db_type = "postgresql"
                print("✅ PostgreSQL cache initialized successfully")
            except Exception as e:
                print(f"⚠️ PostgreSQL connection failed: {e}")
                print("🔄 Falling back to SQLite...")
                self._init_sqlite_fallback()
                self.db_type = "sqlite"
        else:
            self._init_sqlite_fallback()
            self.db_type = "sqlite"
    
    def _init_postgresql(self):
        """Initialize PostgreSQL connection with connection pooling."""
        # Create connection pool
        self.connection_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=self.config.postgres_pool_size,
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            database=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password
        )
        
        # Create tables and indexes
        self._create_postgresql_tables()
    
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables and indexes for caching."""
        with self._get_pg_connection() as conn:
            cursor = conn.cursor()
            
            # Create schema if not exists
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS {self.config.postgres_schema}')
            
            # Cache entries table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config.postgres_schema}.cache_entries (
                    cache_key VARCHAR(64) PRIMARY KEY,
                    cache_type VARCHAR(50) NOT NULL,
                    cached_data BYTEA NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    size_bytes INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 1
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_cache_type 
                ON {self.config.postgres_schema}.cache_entries(cache_type)
            ''')
            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON {self.config.postgres_schema}.cache_entries(expires_at)
            ''')
            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON {self.config.postgres_schema}.cache_entries(created_at)
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_pg_connection(self):
        """Get PostgreSQL connection from pool."""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def _init_sqlite_fallback(self):
        """Initialize SQLite database as fallback."""
        self.db_path = self.config.cache_db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    cache_type TEXT NOT NULL,
                    value BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    accessed_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)')
            conn.commit()
    
    def _generate_cache_key(self, cache_type: str, **kwargs) -> str:
        """Generate a unique cache key based on inputs."""
        key_data = {
            'type': cache_type,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _cleanup_memory_cache(self):
        """Clean up memory cache if it exceeds size limits."""
        current_size_mb = sum(self.cache_sizes.values()) / (1024 * 1024)
        
        if current_size_mb > self.config.max_cache_size_mb:
            # Remove oldest accessed items until we're under the limit
            sorted_items = sorted(
                self.cache_access_times.items(),
                key=lambda x: x[1]
            )
            
            for key, _ in sorted_items:
                if key in self.memory_cache:
                    size = self.cache_sizes.pop(key, 0)
                    del self.memory_cache[key]
                    del self.cache_access_times[key]
                    current_size_mb -= size / (1024 * 1024)
                    
                    if current_size_mb <= self.config.max_cache_size_mb * 0.8:
                        break
    
    def _cleanup_database_cache(self):
        """Clean up expired database entries."""
        if self.db_type == "postgresql":
            with self._get_pg_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    DELETE FROM {self.config.postgres_schema}.cache_entries 
                    WHERE expires_at < NOW()
                ''')
                conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM cache_entries WHERE expires_at < ?', (datetime.now(),))
                conn.commit()
    
    def get(self, cache_type: str, **kwargs) -> Optional[any]:
        """Retrieve item from cache with fallback to database."""
        if not self._is_cache_enabled(cache_type):
            return None
            
        cache_key = self._generate_cache_key(cache_type, **kwargs)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_access_times[cache_key] = time.time()
            self.stats['hits'] += 1
            self.stats['memory_hits'] += 1
            return self.memory_cache[cache_key]
        
        # Check database cache
        if self.db_type == "postgresql":
            return self._get_from_postgresql(cache_key, cache_type)
        else:
            return self._get_from_sqlite(cache_key, cache_type)
    
    def _get_from_postgresql(self, cache_key: str, cache_type: str) -> Optional[any]:
        """Get cached item from PostgreSQL."""
        try:
            with self._get_pg_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    SELECT cached_data, expires_at FROM {self.config.postgres_schema}.cache_entries 
                    WHERE cache_key = %s AND cache_type = %s
                ''', (cache_key, cache_type))
                
                row = cursor.fetchone()
                if row:
                    value_blob, expires_at = row
                    
                    if expires_at is None or expires_at > datetime.now(expires_at.tzinfo):
                        # Valid cache entry - deserialize and store in memory
                        try:
                            value = pickle.loads(bytes(value_blob))
                            
                            # Add to memory cache
                            serialized_size = len(value_blob)
                            self.memory_cache[cache_key] = value
                            self.cache_access_times[cache_key] = time.time()
                            self.cache_sizes[cache_key] = serialized_size
                            
                            # Update access time and count in database
                            cursor.execute(f'''
                                UPDATE {self.config.postgres_schema}.cache_entries 
                                SET last_accessed = NOW(), access_count = access_count + 1
                                WHERE cache_key = %s
                            ''', (cache_key,))
                            conn.commit()
                            
                            self.stats['hits'] += 1
                            self.stats['db_hits'] += 1
                            return value
                        except Exception as e:
                            print(f"Cache deserialization error: {e}")
            
            self.stats['misses'] += 1
            return None
        except Exception as e:
            print(f"PostgreSQL cache retrieval error: {e}")
            self.stats['misses'] += 1
            return None
    
    def _get_from_sqlite(self, cache_key: str, cache_type: str) -> Optional[any]:
        """Get cached item from SQLite (fallback)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT value, expires_at FROM cache_entries 
                    WHERE key = ? AND cache_type = ?
                ''', (cache_key, cache_type))
                
                row = cursor.fetchone()
                if row:
                    value_blob, expires_at = row
                    expires_at = datetime.fromisoformat(expires_at)
                    
                    if expires_at > datetime.now():
                        try:
                            value = pickle.loads(value_blob)
                            
                            # Add to memory cache
                            serialized_size = len(value_blob)
                            self.memory_cache[cache_key] = value
                            self.cache_access_times[cache_key] = time.time()
                            self.cache_sizes[cache_key] = serialized_size
                            
                            # Update access time in database
                            conn.execute('''
                                UPDATE cache_entries SET accessed_at = ? WHERE key = ?
                            ''', (datetime.now(), cache_key))
                            conn.commit()
                            
                            self.stats['hits'] += 1
                            self.stats['db_hits'] += 1
                            return value
                        except Exception as e:
                            print(f"Cache deserialization error: {e}")
            
            self.stats['misses'] += 1
            return None
        except Exception as e:
            print(f"SQLite cache retrieval error: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, cache_type: str, value: any, **kwargs):
        """Store item in both memory and database cache."""
        if not self._is_cache_enabled(cache_type):
            return
            
        cache_key = self._generate_cache_key(cache_type, **kwargs)
        
        try:
            # Serialize the value
            value_blob = pickle.dumps(value)
            serialized_size = len(value_blob)
            
            # Calculate expiry time
            expires_at = datetime.now() + timedelta(hours=self.config.cache_ttl_hours)
            
            # Store in memory cache
            self.memory_cache[cache_key] = value
            self.cache_access_times[cache_key] = time.time()
            self.cache_sizes[cache_key] = serialized_size
            
            # Store in database cache
            if self.db_type == "postgresql":
                self._set_in_postgresql(cache_key, cache_type, value_blob, expires_at, serialized_size, kwargs)
            else:
                self._set_in_sqlite(cache_key, cache_type, value_blob, expires_at, serialized_size)
            
            self.stats['total_cached_items'] += 1
            
            # Cleanup if necessary
            self._cleanup_memory_cache()
            
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    def _set_in_postgresql(self, cache_key: str, cache_type: str, value_blob: bytes, 
                          expires_at: datetime, size_bytes: int, metadata: dict):
        """Store cached item in PostgreSQL."""
        try:
            with self._get_pg_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    INSERT INTO {self.config.postgres_schema}.cache_entries 
                    (cache_key, cache_type, cached_data, expires_at, size_bytes, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        cached_data = EXCLUDED.cached_data,
                        last_accessed = NOW(),
                        expires_at = EXCLUDED.expires_at,
                        size_bytes = EXCLUDED.size_bytes,
                        metadata = EXCLUDED.metadata,
                        access_count = cache_entries.access_count + 1
                ''', (
                    cache_key, cache_type, value_blob, expires_at, size_bytes, json.dumps(metadata)
                ))
                conn.commit()
        except Exception as e:
            print(f"PostgreSQL cache storage error: {e}")
    
    def _set_in_sqlite(self, cache_key: str, cache_type: str, value_blob: bytes, 
                      expires_at: datetime, size_bytes: int):
        """Store cached item in SQLite (fallback)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, cache_type, value, created_at, accessed_at, expires_at, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key, cache_type, value_blob,
                    datetime.now(), datetime.now(), expires_at, size_bytes
                ))
                conn.commit()
        except Exception as e:
            print(f"SQLite cache storage error: {e}")
    
    def _is_cache_enabled(self, cache_type: str) -> bool:
        """Check if caching is enabled for the given type."""
        cache_type_mapping = {
            'response': self.config.enable_response_cache,
            'embedding': self.config.enable_embedding_cache,
            'retrieval': self.config.enable_retrieval_cache,
            'thinking': self.config.enable_thinking_cache
        }
        return cache_type_mapping.get(cache_type, False)
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache entries by type or all if type is None."""
        if cache_type:
            # Clear specific cache type from memory
            keys_to_remove = [
                key for key in self.memory_cache.keys()
                if self._generate_cache_key(cache_type) in key
            ]
            for key in keys_to_remove:
                self.memory_cache.pop(key, None)
                self.cache_access_times.pop(key, None)
                self.cache_sizes.pop(key, None)
            
            # Clear from database
            if self.db_type == "postgresql":
                with self._get_pg_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'''
                        DELETE FROM {self.config.postgres_schema}.cache_entries 
                        WHERE cache_type = %s
                    ''', (cache_type,))
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries WHERE cache_type = ?', (cache_type,))
                    conn.commit()
        else:
            # Clear all cache
            self.memory_cache.clear()
            self.cache_access_times.clear()
            self.cache_sizes.clear()
            
            if self.db_type == "postgresql":
                with self._get_pg_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'DELETE FROM {self.config.postgres_schema}.cache_entries')
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries')
                    conn.commit()
    
    def get_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        # Update memory usage
        self.stats['memory_usage_mb'] = sum(self.cache_sizes.values()) / (1024 * 1024)
        
        # Get database stats
        if self.db_type == "postgresql":
            try:
                with self._get_pg_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'''
                        SELECT COUNT(*), SUM(size_bytes) 
                        FROM {self.config.postgres_schema}.cache_entries
                    ''')
                    db_count, db_size_bytes = cursor.fetchone()
                    
                    cursor.execute(f'''
                        SELECT cache_type, COUNT(*) 
                        FROM {self.config.postgres_schema}.cache_entries 
                        GROUP BY cache_type
                    ''')
                    type_counts = dict(cursor.fetchall())
            except Exception as e:
                print(f"Error getting PostgreSQL stats: {e}")
                db_count, db_size_bytes, type_counts = 0, 0, {}
        else:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('SELECT COUNT(*), SUM(size_bytes) FROM cache_entries')
                    db_count, db_size_bytes = cursor.fetchone()
                    
                    cursor = conn.execute('''
                        SELECT cache_type, COUNT(*) FROM cache_entries 
                        GROUP BY cache_type
                    ''')
                    type_counts = dict(cursor.fetchall())
            except Exception as e:
                print(f"Error getting SQLite stats: {e}")
                db_count, db_size_bytes, type_counts = 0, 0, {}
        
        return {
            **self.stats,
            'database_type': self.db_type,
            'hit_rate': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1),
            'memory_items': len(self.memory_cache),
            'db_items': db_count or 0,
            'db_size_mb': (db_size_bytes or 0) / (1024 * 1024),
            'type_distribution': type_counts
        }

# Initialize cache manager
cache_manager = AdvancedCacheManager(config.cache)

# ── Enhanced Memory Manager ──────────────────────────────────────────────────
class EnhancedMemoryManager:
    """Improved conversation memory with forgetting and importance weighting"""
    def __init__(self, max_tokens: int = 4000):
        self.messages: list = []
        self.max_tokens = max_tokens
        self.token_count: float = 0.0
        self.importance_scores: list = []
    
    def add_message(self, message, importance=1.0):
        """Add message with importance score"""
        self.messages.append(message)
        self.importance_scores.append(importance)
        
        # Estimate tokens (simplified)
        msg_tokens = len(message.content.split()) * 1.3
        self.token_count += msg_tokens
        
        # Prune if needed
        if self.token_count > self.max_tokens:
            self._prune_memory()
    
    def _prune_memory(self):
        """Remove least important messages"""
        # Keep most recent message and most important ones
        if len(self.messages) <= 2:
            return
            
        # Sort by importance (excluding most recent)
        old_msgs = list(zip(self.messages[:-1], self.importance_scores[:-1]))
        sorted_msgs = sorted(old_msgs, key=lambda x: x[1])
        
        # Remove least important
        to_remove = sorted_msgs[0][0]
        idx = self.messages.index(to_remove)
        self.messages.pop(idx)
        self.importance_scores.pop(idx)
        
        # Recalculate tokens
        self.token_count = sum(len(m.content.split()) * 1.3 for m in self.messages)
    
    def get_messages(self):
        """Get current messages"""
        return self.messages

# Create memory manager instance
memory_manager = EnhancedMemoryManager()

def analyze_query_intent(user_content: str) -> dict:
    """Analyze user intent to determine best response strategy."""
    intent_patterns = {
        "calculation": ["calculate", "compute", "what is", "value", "metrics"],
        "analysis": ["analyze", "compare", "evaluate", "assess", "explain"],
        "research": ["search", "find", "lookup", "information about"],
        "regulatory": ["basel", "compliance", "regulation", "requirement"],
        "tool_demo": ["show me", "example", "demo", "how to use"]
    }
    
    user_lower = user_content.lower()
    scores = {}
    
    for intent, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in user_lower)
        if score > 0:
            scores[intent] = score
    
    primary_intent = max(scores.items(), key=lambda x: x[1])[0] if scores else "general"
    
    return {
        "primary_intent": primary_intent,
        "confidence": max(scores.values()) if scores else 0,
        "all_scores": scores
    }

def calculate_response_quality(response: str, user_question: str) -> dict:
    """Calculate comprehensive quality metrics for responses."""
    metrics = {
        "length_score": min(len(response) / 500, 1.0),  # Normalize to 500 chars
        "structure_score": 0.0,
        "source_score": 0.0,
        "completeness_score": 0.0
    }
    
    # Structure scoring
    structure_indicators = ["**", "###", "-", "1.", "2.", "3."]
    metrics["structure_score"] = min(
        sum(1 for indicator in structure_indicators if indicator in response) / 4, 1.0
    )
    
    # Source scoring
    if "**Sources**" in response or "**Source" in response:
        metrics["source_score"] = 1.0
    
    # Completeness scoring (basic keyword matching)
    question_keywords = set(user_question.lower().split())
    response_keywords = set(response.lower().split())
    overlap = len(question_keywords.intersection(response_keywords))
    metrics["completeness_score"] = min(overlap / max(len(question_keywords), 1), 1.0)
    
    # Advanced metrics
    source_count = response.count("- ")  # Count sources
    has_formatting = "###" in response or "##" in response
    has_list = "\n- " in response or "\n1." in response
    has_examples = "example" in response.lower() or "instance" in response.lower()
    complex_reasoning = len(response) > 1000 and "however" in response.lower()
    
    # Calculate specialized scores
    information_density = min(len(re.findall(r'[A-Z][a-z]+', response)) / max(len(response.split()), 1) * 10, 1.0)
    formatting_score = (has_formatting * 0.5) + (has_list * 0.5)
    reasoning_score = 0.5 + (complex_reasoning * 0.5)
    
    # Domain-specific metrics for financial content
    has_financial_terms = bool(re.search(r'ratio|capital|portfolio|risk|compliance|regulatory|basel|var|sharpe', response.lower()))
    has_calculations = bool(re.search(r'\d+[%]|\d+\.\d+', response))
    
    financial_score = (has_financial_terms * 0.6) + (has_calculations * 0.4)
    
    # Combined comprehensive score
    comprehensive_score = (
        metrics["length_score"] * 1.0 +
        metrics["structure_score"] * 2.0 +
        metrics["source_score"] * 1.5 +
        metrics["completeness_score"] * 2.0 +
        formatting_score * 1.5 +
        information_density * 2.0 + 
        reasoning_score * 2.0 +
        financial_score * 3.0
    ) / 15.0
    
    overall_score = comprehensive_score * 10
    
    return {
        **metrics,
        "source_count": source_count,
        "formatting_quality": formatting_score * 10,
        "information_density": information_density * 10,
        "reasoning_complexity": reasoning_score * 10,
        "financial_relevance": financial_score * 10,
        "comprehensive_score": comprehensive_score * 10,
        "overall_score": overall_score
    }

# ── Custom State for Chain-of-Thought with Self-Reflection ─────────────────
class CoTState(TypedDict):
    messages: list
    thinking: Optional[str]
    thinking_quality: Optional[float]
    show_thinking: Optional[bool]
    initial_answer: Optional[str]
    reflection: Optional[str]
    final_answer: Optional[str]
    reflection_score: Optional[float]
    improvement_needed: Optional[bool]
    iteration_count: Optional[int]
    show_reflection: Optional[bool]
    query_intent: Optional[dict]
    response_quality: Optional[dict]
    # Remove chain from state - we'll get it from session instead



# ── Build the RAG retriever once ─────────────────────────────────────────────
INDEX_DIR = (
    Path(__file__).resolve().parent / "vector_store" / "faiss"
)  # backend/vector_store/faiss
_vectordb = FAISS.load_local(
    str(INDEX_DIR),
    OpenAIEmbeddings(),
     allow_dangerous_deserialization=True,
)

_retriever = _vectordb.as_retriever(search_kwargs={"k": 6})

# ── LangGraph workflow ───────────────────────────────────────────────────────
workflow = StateGraph(state_schema=CoTState)

# Use different models for different tasks to optimize costs
thinking_llm = ChatOpenAI(
    model=config.llm.model if not config.reflection.use_cheaper_model else "gpt-4o-mini", 
    temperature=0.3, 
    streaming=True,
    timeout=config.llm.timeout
)
main_llm = ChatOpenAI(
    model=config.llm.model, 
    temperature=config.llm.temperature, 
    streaming=config.llm.streaming,
    timeout=config.llm.timeout
)
reflection_llm = ChatOpenAI(
    model="gpt-4o-mini" if config.reflection.use_cheaper_model else config.llm.model, 
    temperature=0.1, 
    streaming=True,
    timeout=config.llm.timeout
)

# ── Instrumented LLM Calls ───────────────────────────────────────────────────
def instrumented_llm_call(llm, messages, **kwargs):
    """Wrapper for LLM calls with performance monitoring and caching."""
    # Try cache first
    try:
        cached_result = cache_manager.get(
            cache_type='response',
            messages=str(messages),
            model=getattr(llm, 'model_name', str(llm)),
            kwargs=str(kwargs)
        )
        
        if cached_result is not None:
            # Log cache hit
            performance_monitor.log_request(
                model=getattr(llm, 'model_name', str(llm)),
                prompt_tokens=0,
                completion_tokens=0,
                response_time=0.001,  # Very fast cache response
                error=False,
                cache_hit=True
            )
            return cached_result
    except Exception as e:
        print(f"Cache lookup error: {e}")
    
    # Original instrumented call logic
    start_time = time.time()
    error_occurred = False
    
    try:
        response = llm.invoke(messages, **kwargs)
        end_time = time.time()
        
        # Cache the result
        try:
            cache_manager.set(
                cache_type='response',
                value=response,
                messages=str(messages),
                model=getattr(llm, 'model_name', str(llm)),
                kwargs=str(kwargs)
            )
        except Exception as e:
            print(f"Cache storage error: {e}")
        
        # Extract token usage if available
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        else:
            # Estimate tokens
            prompt_tokens = sum(len(m.content.split()) for m in messages) // 0.75
            completion_tokens = len(response.content.split()) // 0.75
        
        # Log the request
        performance_monitor.log_request(
            model=llm.model_name,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            response_time=end_time - start_time,
            error=False
        )
        
        return response
        
    except Exception as e:
        end_time = time.time()
        performance_monitor.log_request(
            model=llm.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            response_time=end_time - start_time,
            error=True
        )
        raise e

async def instrumented_llm_call_async(llm, messages, **kwargs):
    """Async wrapper for LLM calls with performance monitoring and caching."""
    # Try cache first
    try:
        cached_result = cache_manager.get(
            cache_type='response',
            messages=str(messages),
            model=getattr(llm, 'model_name', str(llm)),
            kwargs=str(kwargs)
        )
        
        if cached_result is not None:
            # Log cache hit
            performance_monitor.log_request(
                model=getattr(llm, 'model_name', str(llm)),
                prompt_tokens=0,
                completion_tokens=0,
                response_time=0.001,  # Very fast cache response
                error=False,
                cache_hit=True
            )
            return cached_result
    except Exception as e:
        print(f"Cache lookup error: {e}")
    
    # Original async instrumented call logic
    start_time = time.time()
    
    try:
        response = await llm.ainvoke(messages, **kwargs)
        end_time = time.time()
        
        # Cache the result
        try:
            cache_manager.set(
                cache_type='response',
                value=response,
                messages=str(messages),
                model=getattr(llm, 'model_name', str(llm)),
                kwargs=str(kwargs)
            )
        except Exception as e:
            print(f"Cache storage error: {e}")
        
        # Extract token usage if available
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        else:
            # Estimate tokens
            prompt_tokens = sum(len(m.content.split()) for m in messages) // 0.75
            completion_tokens = len(response.content.split()) // 0.75
        
        # Log the request
        performance_monitor.log_request(
            model=llm.model_name,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            response_time=end_time - start_time,
            error=False
        )
        
        return response
        
    except Exception as e:
        end_time = time.time()
        performance_monitor.log_request(
            model=llm.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            response_time=end_time - start_time,
            error=True
        )
        raise e

async def thinking_node(state: CoTState):
    """Generate enhanced Chain-of-Thought reasoning with structured analysis."""
    last_user_msg = state["messages"][-1]
    user_content = last_user_msg.content
    
    # Analyze query intent
    intent_analysis = analyze_query_intent(user_content)
    
    # Enhanced thinking prompt based on intent
    if intent_analysis["primary_intent"] == "regulatory":
        thinking_prompt = f"""
        Analyze this regulatory/compliance question systematically:
        Question: {user_content}

        Provide structured thinking:
        1. **Regulatory Domain**: [Basel III/CCAR/Dodd-Frank/MiFID/etc.]
        2. **Question Type**: [implementation/compliance/interpretation/assessment]
        3. **Key Requirements**: [main regulatory requirements to address]
        4. **Implementation Approach**: [step-by-step methodology]
        5. **Risk Considerations**: [compliance risks and mitigation]
        6. **Data/Sources Needed**: [regulatory documents, guidelines, precedents]
        """
    elif intent_analysis["primary_intent"] == "analysis":
        thinking_prompt = f"""
        Analyze this financial analysis question systematically:
        Question: {user_content}

        Provide structured thinking:
        1. **Analysis Type**: [portfolio/risk/performance/valuation/market]
        2. **Key Metrics**: [financial ratios, risk measures, performance indicators]
        3. **Methodology**: [analytical framework and approach]
        4. **Data Requirements**: [financial data, market data, benchmarks needed]
        5. **Analysis Steps**: [step-by-step analytical process]
        6. **Expected Insights**: [key insights and conclusions to derive]
        """
    elif intent_analysis["primary_intent"] == "calculation":
        thinking_prompt = f"""
        Analyze this calculation request systematically:
        Question: {user_content}

        Provide structured thinking:
        1. **Calculation Type**: [portfolio metrics/VaR/ratios/valuations]
        2. **Required Inputs**: [specific data points and parameters needed]
        3. **Formula/Method**: [mathematical approach or model]
        4. **Step-by-Step Process**: [calculation sequence]
        5. **Validation Checks**: [reasonableness tests and cross-checks]
        6. **Interpretation Guidelines**: [how to interpret results]
        """
    else:
        # General enhanced thinking prompt
        thinking_prompt = f"""
        Analyze this financial question comprehensively:
        Question: {user_content}

        Provide structured thinking:
        1. **Question Category**: [analysis/calculation/research/regulatory/strategy]
        2. **Core Concepts**: [key financial concepts involved]
        3. **Information Needed**: [data, documents, or context required]
        4. **Analytical Approach**: [methodology and framework]
        5. **Key Considerations**: [important factors and constraints]
        6. **Expected Output**: [format and depth of response needed]
        """
    
    try:
        response = await instrumented_llm_call_async(thinking_llm, [HumanMessage(content=thinking_prompt)])
        state["thinking"] = response.content
        
        # Store thinking quality for later use
        thinking_quality = calculate_response_quality(response.content, user_content)
        state["thinking_quality"] = thinking_quality.get("comprehensive_score", 5.0)
        
    except Exception as e:
        print(f"Error in thinking_node: {e}")
        state["thinking"] = f"Error generating thinking process: {str(e)}"
        state["thinking_quality"] = 1.0
    
    return state

# ── Cached LLM Wrapper Functions ─────────────────────────────────────────────
def cached_llm_call(cache_type: str, llm, messages: list, **kwargs):
    """Cached wrapper for LLM calls with fallback to direct call."""
    try:
        # Check cache first
        cached_result = cache_manager.get(
            cache_type=cache_type,
            messages=str(messages),
            model=getattr(llm, 'model_name', str(llm)),
            **kwargs
        )
        
        if cached_result is not None:
            return cached_result
        
        # Make LLM call if not cached
        result = llm.invoke(messages)
        
        # Cache the result
        cache_manager.set(
            cache_type=cache_type,
            value=result,
            messages=str(messages),
            model=getattr(llm, 'model_name', str(llm)),
            **kwargs
        )
        
        return result
        
    except Exception as e:
        print(f"Error in cached_llm_call: {e}")
        # Fallback to direct call
        return llm.invoke(messages)

def cached_rag_call(query: str, retriever, **kwargs):
    """Cached wrapper for RAG retrieval calls."""
    try:
        # Check cache first
        cached_result = cache_manager.get(
            cache_type='retrieval',
            query=query,
            retriever_config=str(kwargs)
        )
        
        if cached_result is not None:
            return cached_result
        
        # Make retrieval call if not cached
        result = retriever.invoke(query)
        
        # Cache the result
        cache_manager.set(
            cache_type='retrieval',
            value=result,
            query=query,
            retriever_config=str(kwargs)
        )
        
        return result
        
    except Exception as e:
        print(f"Error in cached_rag_call: {e}")
        # Fallback to direct call
        return retriever.invoke(query)

@monitor_performance("RAG with CoT")
def call_rag_with_cot(state: CoTState):
    """Enhanced RAG with Chain-of-Thought integration and quality assessment."""
    try:
        last_user_msg = state["messages"][-1]
        user_content = last_user_msg.content
        thinking = state.get("thinking", "")
        
        # Analyze intent for better routing and store in state
        intent_analysis = analyze_query_intent(user_content)
        state["query_intent"] = intent_analysis
        user_content_lower = user_content.lower()
        
        # Get chain from thread-local storage or session
        chain = getattr(chain_storage, 'chain', None)
        if not chain:
            try:
                chain = cl.user_session.get("chain")
                if chain:
                    chain_storage.chain = chain
            except:
                pass  # Session might not be available in this context
        
        if not chain:
            content = "❌ Error: RAG chain not initialized. Please refresh the page."
            state["initial_answer"] = content
            state["messages"].append(AIMessage(content=content))
            return state
        
        # Intent-based routing with improved prompts
        if intent_analysis["primary_intent"] == "calculation" or any(word in user_content_lower for word in ["calculate portfolio", "portfolio metrics", "sharpe ratio"]):
            content = "🔧 **Portfolio Calculation Tools Available**\n\n" \
                     "To calculate portfolio metrics, please provide:\n" \
                     "- **Portfolio weights** (must sum to 1.0)\n" \
                     "- **Expected returns** for each asset (annual %)\n" \
                     "- **Volatilities** for each asset (annual %)\n" \
                     "- **Correlation matrix** between assets\n\n" \
                     "**Example**: _Calculate metrics for a portfolio with 60% stocks (10% return, 15% volatility) and 40% bonds (4% return, 5% volatility) with 0.3 correlation_\n\n" \
                     "💡 I can calculate: Sharpe Ratio, Portfolio Return, Portfolio Risk, Diversification Benefits"
        
        elif any(word in user_content_lower for word in ["calculate var", "value at risk"]):
            content = "🔧 **Value at Risk (VaR) Calculation Tools**\n\n" \
                     "To calculate VaR, please provide:\n" \
                     "- **Portfolio value** in dollars\n" \
                     "- **Expected annual return** (as percentage)\n" \
                     "- **Annual volatility** (as percentage)\n" \
                     "- **Confidence level** (e.g., 95%, 99%)\n" \
                     "- **Time horizon** in days\n\n" \
                     "**Example**: _Calculate 95% VaR for a $1,000,000 portfolio with 8% expected return and 15% volatility over 1 day_\n\n" \
                     "💡 I can calculate: Historical VaR, Parametric VaR, Monte Carlo VaR"
        
        elif any(word in user_content_lower for word in ["stock price", "ticker", "financial data"]):
            content = "🔧 **Stock Data & Market Analysis Tools**\n\n" \
                     "To get stock information, please specify:\n" \
                     "- **Stock symbol** (e.g., AAPL, MSFT, GOOGL)\n" \
                     "- **Time period** (e.g., 1d, 1w, 1m, 1y, 5y)\n" \
                     "- **Analysis type** (price, returns, volatility, correlations)\n\n" \
                     "**Example**: _Get Apple (AAPL) stock price and performance analysis for the last 6 months_\n\n" \
                     "💡 I can provide: Price charts, Technical indicators, Fundamental ratios, Peer comparisons"
        
        elif intent_analysis["primary_intent"] == "regulatory" or any(word in user_content_lower for word in ["regulatory", "basel", "compliance"]):
            # Enhanced prompt for regulatory queries with structured output
            enhanced_prompt = f"""
            **Regulatory Analysis Context:**
            Thinking Process: {thinking}
            
            **Question:** {user_content}
            
            Please provide a comprehensive regulatory analysis with the following structure:
            
            ## Executive Summary
            [Brief overview of the regulatory topic and key implications]
            
            ## Key Regulatory Requirements
            [Detailed requirements with specific citations and rationale]
            
            ## Implementation Guidance
            [Step-by-step practical implementation approach]
            
            ## Risk Assessment & Mitigation
            [Compliance risks and recommended mitigation strategies]
            
            ## Best Practices & Industry Standards
            [Proven approaches and industry consensus]
            
            ## Recent Updates & Changes
            [Any recent regulatory changes or proposed modifications]
            
            Use clear formatting with headers, bullet points, and specific examples. Include quantitative thresholds where applicable.
            """
            
            try:
                response = instrumented_llm_call(chain, {"input": enhanced_prompt, "chat_history": []}, config={"timeout": 30})
                
                answer = response.get("answer", "No answer generated")
                sources = response.get("context", [])
                
                # Extract unique source names
                unique_sources = set()
                if sources:
                    for doc in sources:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            unique_sources.add(Path(doc.metadata['source']).name)
                
                content = answer + "\n\n**📚 Sources**\n"
                for source in unique_sources:
                    content += f"- {source}\n"
                
                content += "\n\n💡 **Additional Tools**: I also have specialized regulatory tools for compliance checklists and risk assessments!"
                
            except Exception as e:
                print(f"RAG chain error for regulatory query: {e}")
                content = f"❌ I encountered an error while searching regulatory information: {str(e)}\n\nPlease try rephrasing your question or check if the documents are available."
        
        else:
            # Enhanced prompt for general queries with structured approach
            enhanced_prompt = f"""
            **Analysis Context:**
            My structured thinking: {thinking}
            Query Intent: {intent_analysis['primary_intent']} (confidence: {intent_analysis['confidence']})
            
            **Question:** {user_content}
            
            Please provide a comprehensive, well-structured response following these guidelines:
            
            ## Executive Summary
            [Clear, concise overview of the answer]
            
            ## Detailed Analysis
            [In-depth analysis with supporting evidence from sources]
            
            ## Key Insights & Implications
            [Important takeaways and practical implications]
            
            ## Quantitative Analysis
            [Include relevant calculations, ratios, or metrics where applicable]
            
            ## Risk Considerations
            [Potential risks, limitations, or caveats]
            
            ## Actionable Recommendations
            [Specific, practical recommendations]
            
            ## Examples & Case Studies
            [Real-world examples or hypothetical scenarios]
            
            Structure your response to be both comprehensive and accessible to financial professionals. Use proper formatting with headers, bullet points, and numbered lists.
            """
            
            try:
                response = instrumented_llm_call(chain, {"input": enhanced_prompt, "chat_history": []}, config={"timeout": 30})
                
                answer = response.get("answer", "No answer generated")
                sources = response.get("context", [])
                
                # Extract unique source names
                unique_sources = set()
                if sources:
                    for doc in sources:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            unique_sources.add(Path(doc.metadata['source']).name)
                
                content = answer + "\n\n**📚 Sources**\n"
                for source in unique_sources:
                    content += f"- {source}\n"
                    
            except Exception as e:
                print(f"RAG chain error for general query: {e}")
                content = f"❌ I encountered an error while searching for information: {str(e)}\n\nPlease try rephrasing your question or check your connection."
        
        # Calculate response quality metrics
        response_quality = calculate_response_quality(content, user_content)
        state["response_quality"] = response_quality
        
        if config.debug_mode:
            print(f"Response quality score: {response_quality.get('comprehensive_score', 0):.1f}/10")
        
    except Exception as e:
        print(f"Error in call_rag_with_cot: {e}")
        content = f"❌ An unexpected error occurred: {str(e)}"
        # Set minimal quality score for error responses
        state["response_quality"] = {"comprehensive_score": 1.0, "overall_score": 1.0}
    
    # Store initial answer in state for reflection
    state["initial_answer"] = content
    
    # Also add the AI response to the messages for conversation history
    state["messages"].append(AIMessage(content=content))
    
    return state

def reflection_node(state: CoTState):
    """Reflect on the initial answer and identify improvements."""
    print("DEBUG: reflection_node called")
    
    initial_answer = state.get("initial_answer", "")
    # Fix: Get user question from the right place
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    print(f"DEBUG: initial_answer length = {len(initial_answer)}")
    
    reflection_prompt = f"""
    You are an expert financial advisor reviewing your own response. 
    
    Original Question: {user_question}
    
    Your Initial Answer: {initial_answer}
    
    Please critically evaluate this response on:
    1. **Accuracy**: Are all facts and calculations correct?
    2. **Completeness**: Does it address all aspects of the question?
    3. **Clarity**: Is the explanation clear and well-structured?
    4. **Sources**: Are the sources relevant and sufficient?
    5. **Actionability**: Does it provide practical, actionable advice?
    
    Rate the response 1-10 and explain what could be improved.
    
    Format your reflection as:
    **Score**: X/10
    **Strengths**: 
    - [List what works well]
    **Areas for Improvement**:
    - [List specific improvements needed]
    **Recommendation**: [IMPROVE/ACCEPT]
    """
    
    try:
        reflection_response = reflection_llm.invoke([HumanMessage(content=reflection_prompt)])
        
        # Parse the reflection to determine if improvement is needed
        reflection_text = reflection_response.content
        score_match = re.search(r'\*\*Score\*\*:\s*(\d+)', reflection_text)
        score = float(score_match.group(1)) if score_match else 5.0
        
        improvement_needed = "IMPROVE" in reflection_text or score < config.reflection.min_score_threshold
        
        print(f"DEBUG: reflection score = {score}, improvement_needed = {improvement_needed}")
        
        state["reflection"] = reflection_text
        state["reflection_score"] = score
        state["improvement_needed"] = improvement_needed
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
    except Exception as e:
        print(f"Error in reflection_node: {e}")
        state["reflection"] = f"Error during reflection: {str(e)}"
        state["reflection_score"] = 5.0
        state["improvement_needed"] = False
        state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    return state

def improvement_node(state: CoTState):
    """Generate an improved response based on reflection feedback."""
    # Fix: Get user question from the right place
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    initial_answer = state.get("initial_answer", "")
    reflection = state.get("reflection", "")
    thinking = state.get("thinking", "")
    
    improvement_prompt = f"""
    Based on the reflection feedback, provide an improved response.
    
    Original Question: {user_question}
    
    Initial Thinking: {thinking}
    
    Initial Answer: {initial_answer}
    
    Reflection Feedback: {reflection}
    
    Now provide an improved, comprehensive response that addresses the identified weaknesses while maintaining the strengths. Use the same format as before with thinking, structured answer, and sources.
    """
    
    try:
        # Use the chain from thread-local storage or session
        chain = getattr(chain_storage, 'chain', None)
        if not chain:
            try:
                chain = cl.user_session.get("chain")
                if chain:
                    chain_storage.chain = chain
            except:
                pass  # Session might not be available in this context
        
        if not chain:
            # Fallback error handling
            state["final_answer"] = "❌ Error: Cannot improve response - RAG chain not available."
            return state
        
        response = chain.invoke(
            {
                "input": improvement_prompt, 
                "chat_history": []
            },
            config={"timeout": 30}
        )
        
        answer = response.get("answer", "No improved answer generated")
        sources = response.get("context", [])  # New chain returns 'context' instead of 'source_documents'
        
        # Extract unique source names
        unique_sources = set()
        if sources:
            for doc in sources:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    unique_sources.add(Path(doc.metadata['source']).name)
        
        improved_content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            improved_content += f"- {source}\n"
        
        state["final_answer"] = improved_content
        
    except Exception as e:
        print(f"Error in improvement_node: {e}")
        state["final_answer"] = f"❌ Error during improvement: {str(e)}\n\n**Fallback:** Using initial answer.\n\n{initial_answer}"
    
    return state

def finalize_response(state: CoTState):
    """Finalize the response, choosing between initial and improved answers."""
    final_answer = state.get("final_answer") or state.get("initial_answer", "")
    reflection = state.get("reflection", "")
    score = state.get("reflection_score", 0)
    iteration_count = state.get("iteration_count", 0)
    show_reflection = state.get("show_reflection", False)
    
    # Add reflection metadata to the response if reflection was performed and user wants to see it
    if reflection and iteration_count > 0 and show_reflection:
        final_answer += f"\n\n---\n**🔍 Self-Reflection Summary:**\n"
        final_answer += f"- Quality Score: {score}/10\n"
        final_answer += f"- Iterations: {iteration_count}\n"
        if score >= 8:
            final_answer += f"- Status: ✅ High-quality response\n"
        elif score >= 6:
            final_answer += f"- Status: ⚠️ Adequate response\n"
        else:
            final_answer += f"- Status: 🔄 Response improved through reflection\n"
    
    # Update the final message in state
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        state["messages"][-1] = AIMessage(content=final_answer)
    else:
        state["messages"].append(AIMessage(content=final_answer))
    
    return state

def should_reflect(state: CoTState) -> str:
    """Decide whether to reflect on the answer."""
    # Fix: Access the correct message for user content
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return "finalize"
    
    user_content = user_messages[-1].content.lower()
    
    print(f"DEBUG should_reflect: user_content = '{user_content[:50]}...'")
    
    # Trigger reflection for complex financial questions
    complex_keywords = [
        "analyze", "compare", "evaluate", "assess", "strategy", 
        "portfolio", "risk", "regulatory", "compliance", "calculate",
        "framework", "implementation", "factors", "consider"
    ]
    
    is_complex = any(keyword in user_content for keyword in complex_keywords)
    show_reflection = state.get("show_reflection", False)
    
    print(f"DEBUG should_reflect: is_complex = {is_complex}, show_reflection = {show_reflection}")
    
    if is_complex and show_reflection and config.reflection.enabled:
        print("DEBUG should_reflect: returning 'reflect'")
        return "reflect"
    else:
        print("DEBUG should_reflect: returning 'finalize'")
        return "finalize"

def should_improve(state: CoTState) -> str:
    """Decide whether to improve the answer based on reflection."""
    improvement_needed = state.get("improvement_needed", False)
    iteration_count = state.get("iteration_count", 0)
    
    # Use configuration for maximum iterations
    if improvement_needed and iteration_count < config.reflection.max_iterations:
        return "improve"
    else:
        return "finalize"

# Build the workflow with reflection
workflow.add_edge(START, "thinking_node")
workflow.add_edge("thinking_node", "rag_node")
workflow.add_node("thinking_node", thinking_node)
workflow.add_node("rag_node", call_rag_with_cot)
workflow.add_node("reflection_node", reflection_node)
workflow.add_node("improvement_node", improvement_node)
workflow.add_node("finalize_node", finalize_response)

# Add conditional edges for reflection workflow
workflow.add_conditional_edges(
    "rag_node",
    should_reflect,
    {
        "reflect": "reflection_node",
        "finalize": "finalize_node"
    }
)
workflow.add_conditional_edges(
    "reflection_node", 
    should_improve,
    {
        "improve": "improvement_node",
        "finalize": "finalize_node"
    }
)
workflow.add_edge("improvement_node", "finalize_node")
workflow.add_edge("finalize_node", END)

# Memory (optional but kept from your original example)
memory          = MemorySaver()
langgraph_app   = workflow.compile(checkpointer=memory)

# ── Chainlit auth (unchanged) ────────────────────────────────────────────────
@cl.password_auth_callback
def auth_callback(username, password):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin"})
    return None

from rag.qa_chain import make_chain

@cl.on_chat_start
async def on_chat_start():
    """Initialize the enhanced chain when a chat session starts."""
    # Create enhanced chain with default settings
    chain = make_chain(k=config.retrieval.k, retrieval_mode=config.retrieval.mode)
    
    # Store in both session and thread-local storage
    cl.user_session.set("chain", chain)
    chain_storage.chain = chain
    
    # Create enhanced settings UI
    await create_settings_ui()
    
    welcome_message = """🚀 **Quantara-AI Enhanced with Advanced Features!** 

I can help you with:
• **Financial Analysis**: Risk calculations, portfolio metrics, beta analysis
• **Document Search**: Search through regulatory docs, 10-K filings, research papers  
• **Stock Data**: Real-time stock prices and charts
• **Regulatory Compliance**: Basel Framework, compliance checklists, risk assessments

**🧠 Advanced Intelligence Features:**
• **Enhanced Chain-of-Thought**: See my structured analysis process with intent-based reasoning
• **Smart Self-Reflection**: Comprehensive quality evaluation with detailed metrics
• **Adaptive RAG**: Intent-aware retrieval with structured response formatting
• **Performance Monitoring**: Real-time cost and performance tracking
• **Advanced Caching**: Multi-layer caching for ultra-fast responses and cost savings

**📊 Quality & Performance:**
• **Response Quality Scoring**: Automatic evaluation of answer comprehensiveness
• **Cost Optimization**: Smart model selection + intelligent caching (GPT-4 for main tasks, GPT-4o-mini for thinking)
• **Performance Analytics**: Token usage, response times, cache hit rates, and error tracking
• **Multi-Layer Cache**: Memory + SQLite persistence for maximum performance
• **Debug Mode**: View quality metrics and processing details

**� Caching Benefits:**
• **Instant Responses**: Cached answers return in milliseconds
• **Cost Reduction**: No API calls for repeated queries
• **Smart Storage**: Automatic cleanup and size management
• **PostgreSQL Backend**: Enterprise-grade database with connection pooling
• **High Performance**: Advanced indexing and concurrent access support

**�🔧 Enhanced Tools & Capabilities:**
• **Structured Outputs**: Organized responses with clear sections and formatting
• **Intent Recognition**: Automatic query classification for better responses
• **Memory Management**: Intelligent conversation history with importance weighting
• **Error Handling**: Robust error recovery with detailed diagnostics

**💡 Usage Tips:**
- Enable "Chain-of-Thought" to see my thinking process stream in real-time
- Use "Self-Reflection" for quality assessment and improvement suggestions  
- Turn on "Debug Mode" to see quality scores and performance metrics
- Check "Show Performance Statistics" to monitor usage, costs, and cache performance
- Use "Cache Management" to clear cached data when needed

**Available Tools:**
""" + get_tool_info() + """

Ask me anything about finance, risk management, or regulatory compliance - now with enhanced intelligence and monitoring!"""
    
    await cl.Message(welcome_message).send()

async def create_settings_ui():
    """Create enhanced settings UI with performance monitoring."""
    settings = [
        cl.input_widget.Switch(
            id="show_cot",
            label="🧠 Chain-of-Thought Reasoning",
            initial=True
        ),
        cl.input_widget.Switch(
            id="show_reflection",
            label="🔍 Self-Reflection Process",
            initial=False
        ),
        cl.input_widget.Select(
            id="retrieval_mode",
            label="🔍 Retrieval Mode",
            values=["basic", "hybrid", "rerank", "compressed"],
            initial_index=1  # hybrid
        ),
        cl.input_widget.Slider(
            id="rag_k",
            label="📊 Documents to Retrieve",
            initial=config.retrieval.k,
            min=3,
            max=12,
            step=1
        ),
        cl.input_widget.Switch(
            id="debug_mode",
            label="🔧 Debug Mode (Show Quality Metrics)",
            initial=config.debug_mode
        ),
        cl.input_widget.Switch(
            id="show_performance",
            label="📈 Show Performance Statistics",
            initial=False
        ),
        cl.input_widget.Select(
            id="cache_action",
            label="🔄 Cache Management",
            values=["none", "clear_all", "clear_responses", "clear_retrievals"],
            initial_index=0
        )
    ]
    
    await cl.ChatSettings(settings).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates and recreate chain if needed."""
    print(f"Settings updated: {settings}")
    cl.user_session.set("settings", settings)
    
    # Update global debug mode
    config.debug_mode = settings.get("debug_mode", False)
    
    # Check if performance stats should be shown
    show_performance = settings.get("show_performance", False)
    
    # Check if retrieval settings changed
    current_mode = config.retrieval.mode
    current_k = config.retrieval.k
    
    new_mode = settings.get("retrieval_mode", current_mode)
    new_k = settings.get("rag_k", current_k)
    
    # Build settings confirmation message
    settings_msg = f"✅ **Settings Updated:**\n"
    settings_msg += f"- Retrieval mode: `{new_mode}`\n"
    settings_msg += f"- Documents to retrieve: `{new_k}`\n"
    settings_msg += f"- Chain-of-Thought: `{'on' if settings.get('show_cot') else 'off'}`\n"
    settings_msg += f"- Self-Reflection: `{'on' if settings.get('show_reflection') else 'off'}`\n"
    settings_msg += f"- Debug mode: `{'on' if config.debug_mode else 'off'}`"
    
    # Add performance statistics if enabled
    if show_performance:
        stats = performance_monitor.get_statistics()
        if "error" not in stats:
            settings_msg += f"\n\n📈 **Performance Statistics:**\n"
            settings_msg += f"- Total requests: `{stats['total_requests']}`\n"
            settings_msg += f"- Total tokens: `{stats['total_tokens']:,}`\n"
            settings_msg += f"- Estimated cost: `${stats['estimated_cost_usd']}`\n"
            settings_msg += f"- Avg response time: `{stats['avg_response_time_sec']}s`\n"
            settings_msg += f"- Error rate: `{stats['error_rate']*100:.1f}%`\n"
            settings_msg += f"- Avg tokens/request: `{stats['avg_tokens_per_request']}`\n"
            
            if stats['model_usage']:
                settings_msg += f"- Model usage: `{stats['model_usage']}`"
        
        # Add cache statistics
        cache_stats = cache_manager.get_stats()
        settings_msg += f"\n\n🔄 **Cache Statistics:**\n"
        settings_msg += f"- Cache hit rate: `{cache_stats['hit_rate']*100:.1f}%`\n"
        settings_msg += f"- Total hits: `{cache_stats['hits']}`\n"
        settings_msg += f"- Memory items: `{cache_stats['memory_items']}`\n"
        settings_msg += f"- Database items: `{cache_stats['db_items']}`\n"
        settings_msg += f"- Memory usage: `{cache_stats['memory_usage_mb']:.1f} MB`\n"
        settings_msg += f"- Database size: `{cache_stats['db_size_mb']:.1f} MB`"
        
        if cache_stats['type_distribution']:
            settings_msg += f"\n- Cache types: `{cache_stats['type_distribution']}`"
    
    # Handle cache management actions
    cache_action = settings.get("cache_action")
    if cache_action == "clear_all":
        cache_manager.clear_cache()
        settings_msg += f"\n\n🧹 **Cache cleared** - All cached data has been removed."
    elif cache_action == "clear_responses":
        cache_manager.clear_cache("response")
        settings_msg += f"\n\n🧹 **Response cache cleared** - All cached responses have been removed."
    elif cache_action == "clear_retrievals":
        cache_manager.clear_cache("retrieval")
        settings_msg += f"\n\n🧹 **Retrieval cache cleared** - All cached retrievals have been removed."
    
    # Recreate chain if retrieval settings changed
    if new_mode != current_mode or new_k != current_k:
        try:
            chain = make_chain(k=new_k, retrieval_mode=new_mode)
            chain_storage.chain = chain
            cl.user_session.set("chain", chain)
            
            await cl.Message(content=settings_msg).send()
        except Exception as e:
            await cl.Message(
                content=f"⚠️ Error updating retrieval settings: {str(e)}\nUsing previous configuration.\n\n{settings_msg}"
            ).send()
    else:
        # Just update UI settings without recreating chain
        await cl.Message(content=settings_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Process user messages with Chain-of-Thought reasoning and RAG."""
    # Get user settings
    settings = cl.user_session.get("settings", {})
    show_cot = settings.get("show_cot", False)
    show_reflection = settings.get("show_reflection", False)
    
    # Store chain in thread-local storage to avoid serialization issues
    chain = cl.user_session.get("chain")
    if chain:
        chain_storage.chain = chain
    
    # Create a single message that will transform from thinking to answer
    msg = cl.Message(content="🤔 Processing your request...")
    await msg.send()
    
    try:
        # Phase 1: Show thinking process if enabled
        if show_cot:
            await show_thinking_process(msg, message.content)
        
        # Phase 2: Generate and show the answer
        answer = await generate_answer(message.content, show_reflection)
        msg.content = answer["content"]
        
        # Phase 3: Add reflection if enabled and available
        elements = []
        if show_reflection and answer.get("reflection"):
            reflection_accordion = cl.Accordion(
                content=answer["reflection"],
                title="🔍 Self-Reflection Analysis",
                open=config.ui.auto_open_reflection
            )
            elements.append(reflection_accordion)
            msg.elements = elements
        
        await msg.update()
            
    except Exception as e:
        print(f"Error in main message handler: {e}")
        error_content = f"❌ An error occurred while processing your request: {str(e)}"
        msg.content = error_content
        await msg.update()

async def show_thinking_process(msg: cl.Message, user_input: str):
    """Show real-time thinking process with letter-by-letter display."""
    thinking_prompt = f"""
    Analyze this financial question step by step:
    Question: {user_input}

    Provide detailed thinking process:
    1. **Question Analysis**: What type of question is this and what are the key components?
    2. **Required Information**: What data, documents, or knowledge do I need?
    3. **Methodology**: What approach will I use to answer this comprehensively?
    4. **Key Considerations**: What important factors should I keep in mind?
    5. **Expected Outcome**: What kind of response would be most helpful?
    
    Be thorough but concise - aim for 4-6 detailed points.
    """
    
    # Initialize thinking display
    msg.content = """**🧠 Thinking Process:**

```
🤔 Starting analysis...
```

⏳ *Analyzing your question...*"""
    await msg.update()
    
    thinking_content = ""
    try:
        print("Starting thinking process stream...")
        
        # Use astream for real-time token streaming
        stream_started = False
        async for chunk in thinking_llm.astream([HumanMessage(content=thinking_prompt)]):
            if hasattr(chunk, 'content') and chunk.content:
                if not stream_started:
                    print("Stream started, receiving tokens...")
                    stream_started = True
                
                thinking_content += chunk.content
                
                # Update display with current thinking (letter by letter effect)
                formatted_thinking = f"""**🧠 Thinking Process:**

```
{thinking_content}▋
```

⏳ *Analyzing step by step...*"""
                
                msg.content = formatted_thinking
                await msg.update()
                
                # Small delay for visible streaming effect
                await asyncio.sleep(0.03)  # Faster updates for better effect
        
        # Final update without cursor
        final_thinking = f"""**🧠 Thinking Process:**

```
{thinking_content}
```

✅ *Analysis complete - generating response...*"""
        
        msg.content = final_thinking
        await msg.update()
        await asyncio.sleep(0.5)  # Brief pause before showing answer
                
    except Exception as e:
        print(f"Error in thinking stream: {e}")
        # Fallback to non-streaming thinking
        try:
            response = await instrumented_llm_call_async(thinking_llm, [HumanMessage(content=thinking_prompt)])
            thinking_content = response.content
            
            msg.content = f"""**🧠 Thinking Process:**

```
{thinking_content}
```

✅ *Analysis complete - generating response...*"""
            await msg.update()
            await asyncio.sleep(0.5)
        except Exception as fallback_error:
            print(f"Fallback thinking error: {fallback_error}")
            msg.content = f"""**🧠 Thinking Process:**

```
Error generating thinking process: {str(e)}
Fallback error: {str(fallback_error)}
```

⚠️ *Proceeding to answer generation...*"""
            await msg.update()

async def generate_answer(user_input: str, show_reflection: bool = False):
    """Generate the main answer using the RAG chain with enhanced quality assessment."""
    try:
        # Get chain from thread-local storage
        chain = getattr(chain_storage, 'chain', None)
        if not chain:
            chain = cl.user_session.get("chain")
            if chain:
                chain_storage.chain = chain
        
        if not chain:
            return {
                "content": "❌ Error: RAG chain not initialized. Please refresh the page.",
                "reflection": None,
                "quality_score": 1.0
            }
        
        # Analyze query intent for better response structuring
        intent_analysis = analyze_query_intent(user_input)
        
        # Enhanced prompt based on intent
        if intent_analysis["primary_intent"] == "regulatory":
            enhanced_prompt = f"""
            **Regulatory Query Analysis**
            Question: {user_input}
            
            Please provide a structured regulatory analysis:
            
            ## Regulatory Overview
            [Brief context and scope]
            
            ## Key Requirements & Standards
            [Specific regulatory requirements with citations]
            
            ## Implementation Framework
            [Step-by-step implementation approach]
            
            ## Compliance Considerations
            [Risk factors and mitigation strategies]
            
            ## Industry Best Practices
            [Proven approaches and recommendations]
            
            Use professional formatting with clear headers and actionable insights.
            """
        else:
            enhanced_prompt = f"""
            **Financial Analysis Request**
            Query Intent: {intent_analysis['primary_intent']}
            Question: {user_input}
            
            Please provide a comprehensive analysis with:
            
            ## Executive Summary
            [Key takeaways and conclusions]
            
            ## Detailed Analysis
            [In-depth examination with data and evidence]
            
            ## Key Metrics & Calculations
            [Relevant financial ratios, calculations, or quantitative analysis]
            
            ## Risk Assessment
            [Potential risks and considerations]
            
            ## Actionable Recommendations
            [Specific, implementable advice]
            
            Structure your response professionally with clear formatting and source citations.
            """
        
        # Generate answer using enhanced prompt
        start_time = time.time()
        response = chain.invoke(
            {"input": enhanced_prompt, "chat_history": []},
            config={"timeout": 30}
        )
        end_time = time.time()
        
        answer = response.get("answer", "No answer generated")
        sources = response.get("context", [])
        
        # Extract unique source names
        unique_sources = set()
        if sources:
            for doc in sources:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    unique_sources.add(Path(doc.metadata['source']).name)
        
        # Format response with enhanced source presentation
        content = answer + "\n\n" + "─" * 50 + "\n**📚 Sources & References**\n"
        for i, source in enumerate(sorted(unique_sources), 1):
            content += f"{i}. {source}\n"
        
        # Add performance info if debug mode is enabled
        if config.debug_mode:
            response_time = end_time - start_time
            content += f"\n*Debug: Response generated in {response_time:.2f}s*"
        
        # Calculate response quality
        quality_metrics = calculate_response_quality(content, user_input)
        quality_score = quality_metrics.get("comprehensive_score", 5.0)
        
        # Add quality indicator if debug mode
        if config.debug_mode:
            content += f"\n*Quality Score: {quality_score:.1f}/10*"
        
        # Generate enhanced reflection if enabled
        reflection_content = None
        if show_reflection:
            reflection_content = await generate_enhanced_reflection(user_input, content, quality_metrics)
        
        return {
            "content": content,
            "reflection": reflection_content,
            "quality_score": quality_score
        }
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "content": f"❌ An error occurred while generating the answer: {str(e)}\n\nPlease try rephrasing your question or check your connection.",
            "reflection": None,
            "quality_score": 1.0
        }

async def generate_enhanced_reflection(user_question: str, answer: str, quality_metrics: dict):
    """Generate enhanced reflection with detailed quality analysis."""
    reflection_prompt = f"""
    You are an expert financial advisor conducting a comprehensive review of your response.
    
    **Original Question:** {user_question}
    
    **Your Response:** {answer}
    
    **Quality Metrics:**
    - Overall Score: {quality_metrics.get('comprehensive_score', 0):.1f}/10
    - Structure Quality: {quality_metrics.get('formatting_quality', 0):.1f}/10
    - Financial Relevance: {quality_metrics.get('financial_relevance', 0):.1f}/10
    - Information Density: {quality_metrics.get('information_density', 0):.1f}/10
    
    Please provide a detailed critical evaluation:
    
    ## Quality Assessment
    **Overall Score**: X/10
    
    ## Strengths Analysis
    - [List specific strengths with examples]
    - [What worked well in the response]
    - [Areas of particular expertise demonstrated]
    
    ## Areas for Improvement
    - [Specific weaknesses or gaps identified]
    - [Missing information or analysis]
    - [Structural or clarity issues]
    
    ## Content Accuracy Review
    - [Assessment of factual accuracy]
    - [Verification of calculations or claims]
    - [Identification of any potential errors]
    
    ## Professional Standards Check
    - [Adherence to financial industry standards]
    - [Appropriate use of terminology]
    - [Compliance with best practices]
    
    ## Actionability Assessment
    - [How practical and implementable are the recommendations]
    - [Clarity of next steps provided]
    - [Relevance to the user's context]
    
    ## Final Recommendation
    **[ACCEPT/IMPROVE]** - [Brief justification]
    
    ## Improvement Suggestions
    [If IMPROVE, provide specific suggestions for enhancement]
    """
    
    try:
        reflection_response = await instrumented_llm_call_async(reflection_llm, [HumanMessage(content=reflection_prompt)])
        return reflection_response.content
    except Exception as e:
        print(f"Error generating enhanced reflection: {e}")
        return f"Error during reflection: {str(e)}"

async def generate_reflection(user_question: str, answer: str):
    """Generate standard reflection on the answer quality."""
    reflection_prompt = f"""
    You are an expert financial advisor reviewing your own response. 
    
    Original Question: {user_question}
    
    Your Answer: {answer}
    
    Please critically evaluate this response on:
    1. **Accuracy**: Are all facts and calculations correct?
    2. **Completeness**: Does it address all aspects of the question?
    3. **Clarity**: Is the explanation clear and well-structured?
    4. **Sources**: Are the sources relevant and sufficient?
    5. **Actionability**: Does it provide practical, actionable advice?
    
    Rate the response 1-10 and explain what could be improved.
    
    Format your reflection as:
    **Score**: X/10
    **Strengths**: 
    - [List what works well]
    **Areas for Improvement**:
    - [List specific improvements needed]
    """
    
    try:
        reflection_response = await instrumented_llm_call_async(reflection_llm, [HumanMessage(content=reflection_prompt)])
        return reflection_response.content
    except Exception as e:
        print(f"Error generating reflection: {e}")
        return f"Error during reflection: {str(e)}"
