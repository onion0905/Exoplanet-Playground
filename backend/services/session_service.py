"""
Session management service for training sessions
"""
import uuid
import time
from typing import Dict, Any
import threading


class SessionService:
    """Service for managing training sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def create_session(self) -> str:
        """Create a new training session"""
        session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        with self.lock:
            self.sessions[session_id] = {
                'id': session_id,
                'created_at': time.time(),
                'status': 'created',
                'progress': 0,
                'current_step': 'Initializing...',
                'config': {},
                'results': {},
                'error': None
            }
        
        return session_id
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        with self.lock:
            return self.sessions.get(session_id, {})
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session information"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(updates)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.lock:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if current_time - session.get('created_at', 0) > max_age_seconds
            ]
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all sessions"""
        with self.lock:
            return self.sessions.copy()


# Global session service instance
session_service = SessionService()