"""
Session Manager for tracking training/prediction sessions.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading


class SessionManager:
    """Manages training and prediction sessions with state tracking."""
    
    def __init__(self, timeout_minutes: int = 60):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.timeout_minutes = timeout_minutes
        self._lock = threading.Lock()
    
    def create_session(self, session_id: str, session_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            session_type: 'custom_training' or 'pretrained_prediction'
            config: Session configuration (model_type, dataset, etc.)
        
        Returns:
            Session information
        """
        with self._lock:
            session = {
                'session_id': session_id,
                'type': session_type,
                'config': config,
                'status': 'initialized',
                'progress': 0,
                'current_step': 'Initializing...',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'error': None,
                'result': None
            }
            self.sessions[session_id] = session
            return session
    
    def update_progress(self, session_id: str, progress: int, current_step: str, status: str = 'running'):
        """Update session progress."""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].update({
                    'progress': progress,
                    'current_step': current_step,
                    'status': status,
                    'updated_at': datetime.now().isoformat()
                })
    
    def set_result(self, session_id: str, result: Dict[str, Any]):
        """Set session result data."""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['result'] = result
                self.sessions[session_id]['status'] = 'completed'
                self.sessions[session_id]['progress'] = 100
                self.sessions[session_id]['current_step'] = 'Completed'
                self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
    
    def set_error(self, session_id: str, error: str):
        """Set session error."""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['error'] = error
                self.sessions[session_id]['status'] = 'error'
                self.sessions[session_id]['updated_at'] = datetime.now().isoformat()

    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        with self._lock:
            return self.sessions.get(session_id)
    
    def get_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session progress information."""
        session = self.get_session(session_id)
        if session:
            return {
                'progress': session.get('progress', 0),
                'current_step': session.get('current_step', ''),
                'status': session.get('status', 'unknown')
            }
        return None
    
    def get_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session result."""
        session = self.get_session(session_id)
        if session and session.get('status') == 'completed':
            return session.get('result')
        return None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions (optional implementation)."""
        pass
