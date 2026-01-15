"""
Comprehensive authentication and authorization system for AI Code Agent.
"""

import os
import jwt
import bcrypt
import secrets
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import base64
from pathlib import Path
import json

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    print("cryptography package required for security features")
    Fernet = None


class UserRole(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    API_USER = "api_user"
    WEBHOOK = "webhook"


class Permission(Enum):
    """System permissions."""
    READ_ANALYSIS = "read_analysis"
    WRITE_ANALYSIS = "write_analysis"
    MANAGE_REPOSITORIES = "manage_repositories"
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    EXECUTE_WORKFLOWS = "execute_workflows"
    VIEW_METRICS = "view_metrics"
    MANAGE_WEBHOOKS = "manage_webhooks"
    GENERATE_CODE = "generate_code"
    DELETE_DATA = "delete_data"


@dataclass
class User:
    """User entity with authentication details."""
    username: str
    email: str
    role: UserRole
    permissions: List[Permission] = field(default_factory=list)
    password_hash: Optional[str] = None
    api_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: Optional[datetime] = None


@dataclass
class SecurityEvent:
    """Security event for auditing."""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "info"  # info, warning, error, critical


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        if identifier not in self.requests:
            return self.max_requests
        
        # Clean old requests
        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(recent_requests))


class SecurityValidator:
    """Validates security requirements and constraints."""
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength."""
        issues = []
        score = 0
        
        if len(password) >= 8:
            score += 25
        else:
            issues.append("Password must be at least 8 characters long")
        
        if any(c.isupper() for c in password):
            score += 25
        else:
            issues.append("Password must contain uppercase letters")
        
        if any(c.islower() for c in password):
            score += 25
        else:
            issues.append("Password must contain lowercase letters")
        
        if any(c.isdigit() for c in password):
            score += 15
        else:
            issues.append("Password must contain numbers")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 10
        else:
            issues.append("Password should contain special characters")
        
        return {
            "is_valid": len(issues) == 0,
            "score": score,
            "issues": issues,
            "strength": "weak" if score < 50 else "medium" if score < 80 else "strong"
        }
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format."""
        # API keys should be 32+ characters, alphanumeric + special chars
        return (len(api_key) >= 32 and 
                all(c.isalnum() or c in "_-." for c in api_key))
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize user input."""
        # Remove potentially dangerous characters
        sanitized = text.replace("<", "&lt;").replace(">", "&gt;")
        sanitized = sanitized.replace("&", "&amp;").replace('"', "&quot;")
        
        # Truncate to max length
        return sanitized[:max_length]


class EncryptionManager:
    """Handles encryption and decryption of sensitive data."""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv("MASTER_ENCRYPTION_KEY")
        
        if not self.master_key:
            # Generate a new master key
            self.master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            print("Generated new master encryption key. Store it securely!")
            print(f"MASTER_ENCRYPTION_KEY={self.master_key}")
        
        self.fernet = self._create_fernet()
    
    def _create_fernet(self) -> Optional[Fernet]:
        """Create Fernet instance for encryption."""
        if not Fernet:
            return None
        
        key = base64.urlsafe_b64encode(
            hashlib.sha256(self.master_key.encode()).digest()
        )
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.fernet:
            return data  # Return unencrypted if crypto not available
        
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.fernet:
            return encrypted_data  # Return as-is if crypto not available
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception:
            return encrypted_data  # Return original if decryption fails


class AuthenticationManager:
    """Manages user authentication and authorization."""
    
    def __init__(self, jwt_secret: Optional[str] = None):
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.security_events: List[SecurityEvent] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.encryption_manager = EncryptionManager()
        
        # Default rate limiters
        self.rate_limiters["login"] = RateLimiter(max_requests=5, window_seconds=300)  # 5 per 5 min
        self.rate_limiters["api"] = RateLimiter(max_requests=1000, window_seconds=3600)  # 1000 per hour
        self.rate_limiters["webhook"] = RateLimiter(max_requests=10000, window_seconds=3600)  # 10k per hour
        
        # Role-based permissions
        self.role_permissions = {
            UserRole.ADMIN: [p for p in Permission],  # All permissions
            UserRole.DEVELOPER: [
                Permission.READ_ANALYSIS, Permission.WRITE_ANALYSIS,
                Permission.EXECUTE_WORKFLOWS, Permission.GENERATE_CODE,
                Permission.VIEW_METRICS, Permission.MANAGE_REPOSITORIES
            ],
            UserRole.VIEWER: [
                Permission.READ_ANALYSIS, Permission.VIEW_METRICS
            ],
            UserRole.API_USER: [
                Permission.READ_ANALYSIS, Permission.WRITE_ANALYSIS,
                Permission.EXECUTE_WORKFLOWS, Permission.GENERATE_CODE
            ],
            UserRole.WEBHOOK: [
                Permission.WRITE_ANALYSIS, Permission.EXECUTE_WORKFLOWS
            ]
        }
        
        # Create default admin user if none exists
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if none exists."""
        admin_users = [u for u in self.users.values() if u.role == UserRole.ADMIN]
        if not admin_users:
            default_password = os.getenv("ADMIN_PASSWORD") or "admin123!"
            admin_user = self.create_user(
                username="admin",
                email="admin@example.com",
                password=default_password,
                role=UserRole.ADMIN
            )
            if admin_user:
                print("Created default admin user:")
                print(f"Username: admin")
                print(f"Password: {default_password}")
                print("IMPORTANT: Change the default password immediately!")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return "ak_" + secrets.token_urlsafe(32)
    
    def create_user(self, username: str, email: str, password: str, 
                   role: UserRole) -> Optional[User]:
        """Create a new user."""
        # Validate input
        username = SecurityValidator.sanitize_input(username, 50)
        email = SecurityValidator.sanitize_input(email, 100)
        
        if username in self.users:
            return None
        
        # Validate password strength
        password_validation = SecurityValidator.validate_password_strength(password)
        if not password_validation["is_valid"]:
            raise ValueError(f"Password validation failed: {password_validation['issues']}")
        
        # Create user
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions.get(role, []),
            password_hash=self.hash_password(password),
            api_key=self.generate_api_key()
        )
        
        self.users[username] = user
        
        # Log security event
        self._log_security_event(
            "user_created",
            user_id=username,
            ip_address="system",
            user_agent="system",
            details={"role": role.value}
        )
        
        return user
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate user with username/password."""
        # Rate limiting for login attempts
        rate_key = f"login:{ip_address}"
        if not self.rate_limiters["login"].is_allowed(rate_key):
            self._log_security_event(
                "login_rate_limited",
                user_id=username,
                ip_address=ip_address,
                user_agent=user_agent,
                details={},
                severity="warning"
            )
            raise Exception("Too many login attempts. Please try again later.")
        
        # Check if user exists
        user = self.users.get(username)
        if not user or not user.is_active:
            self._log_security_event(
                "login_failed_invalid_user",
                user_id=username,
                ip_address=ip_address,
                user_agent=user_agent,
                details={},
                severity="warning"
            )
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_security_event(
                "login_failed_account_locked",
                user_id=username,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"locked_until": user.locked_until.isoformat()},
                severity="warning"
            )
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(minutes=30)
                self._log_security_event(
                    "account_locked",
                    user_id=username,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={"failed_attempts": user.failed_login_attempts},
                    severity="error"
                )
            
            self._log_security_event(
                "login_failed_invalid_password",
                user_id=username,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"failed_attempts": user.failed_login_attempts},
                severity="warning"
            )
            return None
        
        # Successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Create session
        session_id = self._create_session(user, ip_address, user_agent)
        
        self._log_security_event(
            "login_successful",
            user_id=username,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"session_id": session_id}
        )
        
        return session_id
    
    def authenticate_api_key(self, api_key: str, ip_address: str) -> Optional[User]:
        """Authenticate using API key."""
        # Rate limiting for API requests
        rate_key = f"api:{ip_address}"
        if not self.rate_limiters["api"].is_allowed(rate_key):
            self._log_security_event(
                "api_rate_limited",
                user_id=None,
                ip_address=ip_address,
                user_agent="api",
                details={"api_key_prefix": api_key[:8] + "..."},
                severity="warning"
            )
            raise Exception("API rate limit exceeded")
        
        # Validate API key format
        if not SecurityValidator.validate_api_key_format(api_key):
            self._log_security_event(
                "api_auth_invalid_format",
                user_id=None,
                ip_address=ip_address,
                user_agent="api",
                details={"api_key_prefix": api_key[:8] + "..."},
                severity="warning"
            )
            return None
        
        # Find user with matching API key
        for user in self.users.values():
            if user.api_key == api_key and user.is_active:
                self._log_security_event(
                    "api_auth_successful",
                    user_id=user.username,
                    ip_address=ip_address,
                    user_agent="api",
                    details={}
                )
                return user
        
        self._log_security_event(
            "api_auth_failed",
            user_id=None,
            ip_address=ip_address,
            user_agent="api",
            details={"api_key_prefix": api_key[:8] + "..."},
            severity="warning"
        )
        return None
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None
            
            # Check if session is still active
            session_id = payload.get("session_id")
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_active and datetime.now() < session.expires_at:
                    session.last_activity = datetime.now()
                    return payload
            
            return None
            
        except jwt.InvalidTokenError:
            return None
    
    def _create_session(self, user: User, ip_address: str, user_agent: str) -> str:
        """Create user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        session = Session(
            session_id=session_id,
            user_id=user.username,
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            last_activity=datetime.now()
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def create_jwt_token(self, user: User, session_id: str) -> str:
        """Create JWT token for user."""
        payload = {
            "user_id": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "session_id": session_id,
            "iat": time.time(),
            "exp": time.time() + 86400  # 24 hours
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions
    
    def require_permission(self, user: User, permission: Permission):
        """Require user to have specific permission (raises exception if not)."""
        if not self.has_permission(user, permission):
            raise PermissionError(f"User {user.username} does not have permission {permission.value}")
    
    def logout_user(self, session_id: str, ip_address: str):
        """Logout user by invalidating session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            
            self._log_security_event(
                "logout",
                user_id=session.user_id,
                ip_address=ip_address,
                user_agent="",
                details={"session_id": session_id}
            )
    
    def revoke_api_key(self, username: str) -> bool:
        """Revoke user's API key."""
        if username in self.users:
            old_key = self.users[username].api_key
            self.users[username].api_key = self.generate_api_key()
            
            self._log_security_event(
                "api_key_revoked",
                user_id=username,
                ip_address="system",
                user_agent="system",
                details={"old_key_prefix": old_key[:8] + "..." if old_key else "none"}
            )
            return True
        return False
    
    def _log_security_event(self, event_type: str, user_id: Optional[str],
                           ip_address: str, user_agent: str, 
                           details: Dict[str, Any], severity: str = "info"):
        """Log security event for auditing."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        # Log critical events
        if severity in ["error", "critical"]:
            print(f"SECURITY EVENT [{severity.upper()}]: {event_type} - {details}")
    
    def get_security_events(self, user_id: Optional[str] = None, 
                           event_type: Optional[str] = None,
                           severity: Optional[str] = None,
                           limit: int = 100) -> List[SecurityEvent]:
        """Get security events with filtering."""
        events = self.security_events
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_user_sessions(self, username: str) -> List[Session]:
        """Get active sessions for user."""
        return [s for s in self.sessions.values() 
                if s.user_id == username and s.is_active]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if now > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary and statistics."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [e for e in self.security_events if e.timestamp > last_24h]
        
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "active_sessions": len([s for s in self.sessions.values() if s.is_active]),
            "events_last_24h": len(recent_events),
            "failed_logins_last_24h": len([e for e in recent_events if e.event_type == "login_failed_invalid_password"]),
            "api_requests_last_24h": len([e for e in recent_events if e.event_type == "api_auth_successful"]),
            "locked_accounts": len([u for u in self.users.values() if u.locked_until and now < u.locked_until]),
            "security_events_by_severity": {
                "critical": len([e for e in recent_events if e.severity == "critical"]),
                "error": len([e for e in recent_events if e.severity == "error"]),
                "warning": len([e for e in recent_events if e.severity == "warning"]),
                "info": len([e for e in recent_events if e.severity == "info"])
            }
        }


# Global authentication manager
_auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """Get or create global authentication manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager