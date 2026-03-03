import os
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict

logger = logging.getLogger(__name__)

PLAN_MEMORY_DB_PATH = os.environ.get("PLAN_MEMORY_DB_PATH", "./vetinari_memory.db")
PLAN_RETENTION_DAYS = int(os.environ.get("PLAN_RETENTION_DAYS", "90"))
PLAN_ADMIN_TOKEN = os.environ.get("PLAN_ADMIN_TOKEN", "")


class MemoryStore:
    """SQLite-backed memory store for plan histories and subtask outcomes.
    
    Primary storage: SQLite (ACID, indexing, concurrent access)
    Fallback: JSON-on-disk (for development or when SQLite unavailable)
    """
    
    def __init__(self, db_path: str = PLAN_MEMORY_DB_PATH, use_json_fallback: bool = False):
        self.db_path = db_path
        self.use_json_fallback = use_json_fallback
        self._conn = None
        self._json_path = db_path.replace(".db", ".json")
        
        if use_json_fallback:
            self._init_json_store()
        else:
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database with required tables."""
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            
            cursor = self._conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PlanHistory (
                    plan_id TEXT PRIMARY KEY,
                    plan_version INTEGER DEFAULT 1,
                    goal TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'draft',
                    plan_json TEXT,
                    plan_explanation_json TEXT,
                    chosen_plan_id TEXT,
                    plan_justification TEXT,
                    risk_score REAL DEFAULT 0.0,
                    dry_run BOOLEAN DEFAULT 0,
                    auto_approved BOOLEAN DEFAULT 0
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS SubtaskMemory (
                    subtask_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    parent_subtask_id TEXT,
                    description TEXT NOT NULL,
                    depth INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    assigned_model_id TEXT,
                    outcome TEXT,
                    duration_seconds REAL,
                    cost_estimate REAL,
                    rationale TEXT,
                    subtask_explanation_json TEXT,
                    domain TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (plan_id) REFERENCES PlanHistory(plan_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ModelPerformance (
                    model_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    avg_latency REAL DEFAULT 0.0,
                    total_uses INTEGER DEFAULT 0,
                    last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_id, task_type)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_subtask_plan_id ON SubtaskMemory(plan_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_created_at ON PlanHistory(created_at)
            """)
            
            self._conn.commit()
            logger.info(f"Memory store initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            logger.warning(f"SQLite initialization failed: {e}. Falling back to JSON.")
            self._conn = None
            self._init_json_store()
    
    def _init_json_store(self):
        """Initialize JSON fallback store."""
        self.use_json_fallback = True
        if not os.path.exists(self._json_path):
            self._json_data = {
                "plans": {},
                "subtasks": {},
                "model_performance": {}
            }
            self._save_json()
        else:
            with open(self._json_path, 'r') as f:
                self._json_data = json.load(f)
        logger.info(f"JSON fallback memory store initialized at {self._json_path}")
    
    def _save_json(self):
        """Save JSON data to disk."""
        with open(self._json_path, 'w') as f:
            json.dump(self._json_data, f, indent=2, default=str)
    
    def write_plan_history(self, plan_data: Dict[str, Any]) -> bool:
        """Write or update a plan in the memory store."""
        if self.use_json_fallback:
            return self._write_plan_json(plan_data)
        
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO PlanHistory 
                (plan_id, plan_version, goal, updated_at, status, plan_json, plan_explanation_json,
                 chosen_plan_id, plan_justification, risk_score, dry_run, auto_approved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan_data.get("plan_id"),
                plan_data.get("plan_version", 1),
                plan_data.get("goal"),
                datetime.now().isoformat(),
                plan_data.get("status", "draft"),
                json.dumps(plan_data.get("plan_json", {})),
                plan_data.get("plan_explanation_json", ""),
                plan_data.get("chosen_plan_id"),
                plan_data.get("plan_justification"),
                plan_data.get("risk_score", 0.0),
                plan_data.get("dry_run", False),
                plan_data.get("auto_approved", False)
            ))
            self._conn.commit()
            logger.info(f"Plan {plan_data.get('plan_id')} written to memory store")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to write plan: {e}")
            return False
    
    def _write_plan_json(self, plan_data: Dict[str, Any]) -> bool:
        """Write plan to JSON fallback."""
        plan_id = plan_data.get("plan_id")
        self._json_data["plans"][plan_id] = {
            **plan_data,
            "updated_at": datetime.now().isoformat()
        }
        self._save_json()
        return True
    
    def write_subtask_memory(self, subtask_data: Dict[str, Any]) -> bool:
        """Write or update a subtask in the memory store."""
        if self.use_json_fallback:
            return self._write_subtask_json(subtask_data)
        
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO SubtaskMemory
                (subtask_id, plan_id, parent_subtask_id, description, depth,
                 status, assigned_model_id, outcome, duration_seconds, 
                 cost_estimate, rationale, subtask_explanation_json, domain, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                subtask_data.get("subtask_id"),
                subtask_data.get("plan_id"),
                subtask_data.get("parent_subtask_id"),
                subtask_data.get("description"),
                subtask_data.get("depth", 0),
                subtask_data.get("status", "pending"),
                subtask_data.get("assigned_model_id"),
                subtask_data.get("outcome"),
                subtask_data.get("duration_seconds"),
                subtask_data.get("cost_estimate"),
                subtask_data.get("rationale"),
                subtask_data.get("subtask_explanation_json", ""),
                subtask_data.get("domain"),
                datetime.now().isoformat()
            ))
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to write subtask: {e}")
            return False
    
    def _write_subtask_json(self, subtask_data: Dict[str, Any]) -> bool:
        """Write subtask to JSON fallback."""
        subtask_id = subtask_data.get("subtask_id")
        self._json_data["subtasks"][subtask_id] = {
            **subtask_data,
            "updated_at": datetime.now().isoformat()
        }
        self._save_json()
        return True
    
    def query_plan_history(self, plan_id: Optional[str] = None, 
                          goal_contains: Optional[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Query plan histories from memory store."""
        if self.use_json_fallback:
            return self._query_plan_json(plan_id, goal_contains, limit)
        
        try:
            cursor = self._conn.cursor()
            
            if plan_id:
                cursor.execute("""
                    SELECT * FROM PlanHistory 
                    WHERE plan_id = ? 
                    ORDER BY created_at DESC
                """, (plan_id,))
            elif goal_contains:
                cursor.execute("""
                    SELECT * FROM PlanHistory 
                    WHERE goal LIKE ? 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (f"%{goal_contains}%", limit))
            else:
                cursor.execute("""
                    SELECT * FROM PlanHistory 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Failed to query plans: {e}")
            return []
    
    def _query_plan_json(self, plan_id: Optional[str], 
                         goal_contains: Optional[str], limit: int) -> List[Dict]:
        """Query plans from JSON fallback."""
        plans = list(self._json_data["plans"].values())
        
        if plan_id:
            plans = [p for p in plans if p.get("plan_id") == plan_id]
        elif goal_contains:
            plans = [p for p in plans if goal_contains.lower() in p.get("goal", "").lower()]
        
        plans.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return plans[:limit]
    
    def query_subtasks(self, plan_id: Optional[str] = None,
                      subtask_id: Optional[str] = None,
                      depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query subtasks from memory store."""
        if self.use_json_fallback:
            return self._query_subtasks_json(plan_id, subtask_id, depth)
        
        try:
            cursor = self._conn.cursor()
            
            if subtask_id:
                cursor.execute("""
                    SELECT * FROM SubtaskMemory 
                    WHERE subtask_id = ?
                """, (subtask_id,))
            elif plan_id:
                if depth is not None:
                    cursor.execute("""
                        SELECT * FROM SubtaskMemory 
                        WHERE plan_id = ? AND depth = ?
                        ORDER BY depth, subtask_id
                    """, (plan_id, depth))
                else:
                    cursor.execute("""
                        SELECT * FROM SubtaskMemory 
                        WHERE plan_id = ?
                        ORDER BY depth, subtask_id
                    """, (plan_id,))
            else:
                cursor.execute("""
                    SELECT * FROM SubtaskMemory 
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Failed to query subtasks: {e}")
            return []
    
    def _query_subtasks_json(self, plan_id: Optional[str],
                            subtask_id: Optional[str], 
                            depth: Optional[int]) -> List[Dict]:
        """Query subtasks from JSON fallback."""
        subtasks = list(self._json_data["subtasks"].values())
        
        if subtask_id:
            subtasks = [s for s in subtasks if s.get("subtask_id") == subtask_id]
        elif plan_id:
            subtasks = [s for s in subtasks if s.get("plan_id") == plan_id]
            if depth is not None:
                subtasks = [s for s in subtasks if s.get("depth") == depth]
        
        return subtasks
    
    def update_model_performance(self, model_id: str, task_type: str,
                                success: bool, latency: float) -> bool:
        """Update model performance metrics."""
        if self.use_json_fallback:
            return self._update_model_perf_json(model_id, task_type, success, latency)
        
        try:
            cursor = self._conn.cursor()
            
            cursor.execute("""
                SELECT * FROM ModelPerformance 
                WHERE model_id = ? AND task_type = ?
            """, (model_id, task_type))
            
            row = cursor.fetchone()
            
            if row:
                new_success_rate = (row["success_rate"] * row["total_uses"] + (1 if success else 0)) / (row["total_uses"] + 1)
                new_latency = (row["avg_latency"] * row["total_uses"] + latency) / (row["total_uses"] + 1)
                new_uses = row["total_uses"] + 1
                
                cursor.execute("""
                    UPDATE ModelPerformance
                    SET success_rate = ?, avg_latency = ?, 
                        total_uses = ?, last_used_at = ?
                    WHERE model_id = ? AND task_type = ?
                """, (new_success_rate, new_latency, new_uses,
                      datetime.now().isoformat(), model_id, task_type))
            else:
                cursor.execute("""
                    INSERT INTO ModelPerformance
                    (model_id, task_type, success_rate, avg_latency, total_uses, last_used_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (model_id, task_type, 1.0 if success else 0.0, 
                      latency, datetime.now().isoformat()))
            
            self._conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to update model performance: {e}")
            return False
    
    def _update_model_perf_json(self, model_id: str, task_type: str,
                                success: bool, latency: float) -> bool:
        """Update model performance in JSON fallback."""
        key = f"{model_id}:{task_type}"
        if key not in self._json_data["model_performance"]:
            self._json_data["model_performance"][key] = {
                "model_id": model_id,
                "task_type": task_type,
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "total_uses": 0
            }
        
        perf = self._json_data["model_performance"][key]
        total = perf["total_uses"] + 1
        perf["success_rate"] = (perf["success_rate"] * perf["total_uses"] + (1 if success else 0)) / total
        perf["avg_latency"] = (perf["avg_latency"] * perf["total_uses"] + latency) / total
        perf["total_uses"] = total
        perf["last_used_at"] = datetime.now().isoformat()
        
        self._save_json()
        return True
    
    def prune_old_plans(self, retention_days: int = PLAN_RETENTION_DAYS) -> int:
        """Prune plans older than retention_days. Returns count of deleted plans."""
        if self.use_json_fallback:
            return self._prune_old_json(retention_days)
        
        try:
            cutoff = datetime.now() - timedelta(days=retention_days)
            cursor = self._conn.cursor()
            
            cursor.execute("""
                SELECT plan_id FROM PlanHistory 
                WHERE created_at < ?
            """, (cutoff.isoformat(),))
            
            old_plans = [row[0] for row in cursor.fetchall()]
            
            for plan_id in old_plans:
                cursor.execute("DELETE FROM SubtaskMemory WHERE plan_id = ?", (plan_id,))
                cursor.execute("DELETE FROM PlanHistory WHERE plan_id = ?", (plan_id,))
            
            self._conn.commit()
            logger.info(f"Pruned {len(old_plans)} old plans")
            return len(old_plans)
            
        except sqlite3.Error as e:
            logger.error(f"Failed to prune old plans: {e}")
            return 0
    
    def _prune_old_json(self, retention_days: int) -> int:
        """Prune old plans from JSON fallback."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff.isoformat()
        
        to_delete = [
            pid for pid, p in self._json_data["plans"].items()
            if p.get("created_at", "") < cutoff_str
        ]
        
        for pid in to_delete:
            del self._json_data["plans"][pid]
            self._json_data["subtasks"] = {
                sid: s for sid, s in self._json_data["subtasks"].items()
                if s.get("plan_id") != pid
            }
        
        self._save_json()
        return len(to_delete)
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        if self.use_json_fallback:
            return {
                "total_plans": len(self._json_data["plans"]),
                "total_subtasks": len(self._json_data["subtasks"]),
                "total_model_records": len(self._json_data["model_performance"]),
                "storage_type": "json"
            }
        
        try:
            cursor = self._conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM PlanHistory")
            plan_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM SubtaskMemory")
            subtask_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ModelPerformance")
            model_count = cursor.fetchone()[0]
            
            return {
                "total_plans": plan_count,
                "total_subtasks": subtask_count,
                "total_model_records": model_count,
                "storage_type": "sqlite",
                "db_path": self.db_path
            }
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}


_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store instance."""
    global _memory_store
    if _memory_store is None:
        use_json = os.environ.get("PLAN_USE_JSON_FALLBACK", "false").lower() in ("1", "true", "yes")
        _memory_store = MemoryStore(use_json_fallback=use_json)
    return _memory_store


def init_memory_store(db_path: str = None, use_json_fallback: bool = False) -> MemoryStore:
    """Initialize a new memory store instance."""
    global _memory_store
    if db_path:
        _memory_store = MemoryStore(db_path=db_path, use_json_fallback=use_json_fallback)
    else:
        _memory_store = MemoryStore(use_json_fallback=use_json_fallback)
    return _memory_store


try:
    from .memory import (
        DualMemoryStore,
        get_dual_memory_store,
        init_dual_memory_store,
        MemoryEntry,
        MemoryEntryType,
        ApprovalDetails,
        MEMORY_BACKEND_MODE
    )
    DUAL_MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dual memory backends not available: {e}")
    DualMemoryStore = None
    get_dual_memory_store = None
    init_dual_memory_store = None
    MemoryEntry = None
    MemoryEntryType = None
    ApprovalDetails = None
    MEMORY_BACKEND_MODE = None
    DUAL_MEMORY_AVAILABLE = False
