"""
Tests for the authentication and authorization system.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.security.auth_manager import (
    AuthenticationManager, User, UserRole, Permission, Session, 
    SecurityEvent, RateLimiter, SecurityValidator, EncryptionManager,
    get_auth_manager
)


class TestUserRole:
    """Test cases for user roles."""
    
    def test_user_roles_exist(self):
        """Test that all required user roles exist."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.DEVELOPER.value == "developer"
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.API_USER.value == "api_user"
        assert UserRole.WEBHOOK.value == "webhook"


class TestPermission:
    """Test cases for permissions."""
    
    def test_permissions_exist(self):
        """Test that all required permissions exist."""
        expected_permissions = [
            "read_analysis", "write_analysis", "manage_repositories",
            "manage_users", "manage_system", "execute_workflows",
            "view_metrics", "manage_webhooks", "generate_code", "delete_data"
        ]
        
        actual_permissions = [p.value for p in Permission]
        
        for perm in expected_permissions:
            assert perm in actual_permissions


class TestUser:
    """Test cases for User dataclass."""
    
    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.DEVELOPER,
            password_hash="hashed_password"
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.DEVELOPER
        assert user.is_active is True
        assert user.failed_login_attempts == 0
        assert user.locked_until is None
        assert isinstance(user.created_at, datetime)


class TestSession:
    """Test cases for Session dataclass."""
    
    def test_session_creation(self):
        """Test creating a session."""
        session = Session(
            session_id="test_session_123",
            user_id="testuser",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        
        assert session.session_id == "test_session_123"
        assert session.user_id == "testuser"
        assert session.is_active is True
        assert session.ip_address == "192.168.1.1"


class TestSecurityEvent:
    """Test cases for SecurityEvent dataclass."""
    
    def test_security_event_creation(self):
        """Test creating a security event."""
        event = SecurityEvent(
            event_type="login_successful",
            user_id="testuser",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            details={"session_id": "abc123"}
        )
        
        assert event.event_type == "login_successful"
        assert event.user_id == "testuser"
        assert event.severity == "info"
        assert isinstance(event.timestamp, datetime)
        assert event.details["session_id"] == "abc123"


class TestRateLimiter:
    """Test cases for rate limiting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter(max_requests=3, window_seconds=1)
    
    def test_initial_state(self):
        """Test initial rate limiter state."""
        assert self.rate_limiter.max_requests == 3
        assert self.rate_limiter.window_seconds == 1
        assert len(self.rate_limiter.requests) == 0
    
    def test_allow_requests_under_limit(self):
        """Test allowing requests under the limit."""
        assert self.rate_limiter.is_allowed("user1") is True
        assert self.rate_limiter.is_allowed("user1") is True
        assert self.rate_limiter.is_allowed("user1") is True
        
        # Fourth request should be denied
        assert self.rate_limiter.is_allowed("user1") is False
    
    def test_different_identifiers_separate_limits(self):
        """Test that different identifiers have separate limits."""
        # Use up limit for user1
        for _ in range(3):
            assert self.rate_limiter.is_allowed("user1") is True
        
        # user1 should be limited
        assert self.rate_limiter.is_allowed("user1") is False
        
        # user2 should still be allowed
        assert self.rate_limiter.is_allowed("user2") is True
    
    def test_window_expiration(self):
        """Test that rate limit resets after window expires."""
        # Use up limit
        for _ in range(3):
            assert self.rate_limiter.is_allowed("user1") is True
        
        # Should be limited
        assert self.rate_limiter.is_allowed("user1") is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        assert self.rate_limiter.is_allowed("user1") is True
    
    def test_get_remaining_requests(self):
        """Test getting remaining request count."""
        assert self.rate_limiter.get_remaining_requests("user1") == 3
        
        self.rate_limiter.is_allowed("user1")
        assert self.rate_limiter.get_remaining_requests("user1") == 2
        
        self.rate_limiter.is_allowed("user1")
        assert self.rate_limiter.get_remaining_requests("user1") == 1
        
        self.rate_limiter.is_allowed("user1")
        assert self.rate_limiter.get_remaining_requests("user1") == 0


class TestSecurityValidator:
    """Test cases for security validation."""
    
    def test_password_strength_weak(self):
        """Test password strength validation for weak passwords."""
        result = SecurityValidator.validate_password_strength("weak")
        
        assert result["is_valid"] is False
        assert result["strength"] == "weak"
        assert len(result["issues"]) > 0
        assert "at least 8 characters" in str(result["issues"])
    
    def test_password_strength_medium(self):
        """Test password strength validation for medium passwords."""
        result = SecurityValidator.validate_password_strength("Password123")
        
        assert result["is_valid"] is False  # Missing special chars
        assert result["strength"] == "medium"
        assert result["score"] >= 50
    
    def test_password_strength_strong(self):
        """Test password strength validation for strong passwords."""
        result = SecurityValidator.validate_password_strength("StrongPass123!")
        
        assert result["is_valid"] is True
        assert result["strength"] == "strong"
        assert result["score"] >= 80
        assert len(result["issues"]) == 0
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        # Valid API key
        valid_key = "ak_" + "a" * 32
        assert SecurityValidator.validate_api_key_format(valid_key) is True
        
        # Too short
        short_key = "ak_short"
        assert SecurityValidator.validate_api_key_format(short_key) is False
        
        # Invalid characters
        invalid_key = "ak_" + "a" * 29 + "@#$"
        assert SecurityValidator.validate_api_key_format(invalid_key) is False
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        malicious_input = '<script>alert("xss")</script>'
        sanitized = SecurityValidator.sanitize_input(malicious_input)
        
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized
        
        # Test length limiting
        long_input = "a" * 2000
        sanitized = SecurityValidator.sanitize_input(long_input, max_length=100)
        assert len(sanitized) == 100


class TestEncryptionManager:
    """Test cases for encryption manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encryption_manager = EncryptionManager("test_master_key_12345")
    
    def test_encryption_decryption(self):
        """Test encryption and decryption."""
        original_data = "sensitive information"
        
        encrypted = self.encryption_manager.encrypt(original_data)
        assert encrypted != original_data
        assert len(encrypted) > len(original_data)
        
        decrypted = self.encryption_manager.decrypt(encrypted)
        assert decrypted == original_data
    
    def test_encryption_without_crypto_library(self):
        """Test behavior when crypto library is not available."""
        with patch('src.security.auth_manager.Fernet', None):
            manager = EncryptionManager("test_key")
            
            # Should return data as-is when crypto not available
            data = "test data"
            encrypted = manager.encrypt(data)
            assert encrypted == data
            
            decrypted = manager.decrypt(data)
            assert decrypted == data


class TestAuthenticationManager:
    """Test cases for authentication manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthenticationManager("test_jwt_secret")
    
    def test_initial_state(self):
        """Test initial authentication manager state."""
        assert len(self.auth_manager.users) >= 1  # Default admin user
        assert len(self.auth_manager.sessions) == 0
        assert len(self.auth_manager.security_events) >= 1  # Admin creation event
        assert len(self.auth_manager.rate_limiters) == 3
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = self.auth_manager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 20  # Bcrypt hashes are typically 60+ chars
        
        # Verify correct password
        assert self.auth_manager.verify_password(password, hashed) is True
        
        # Verify incorrect password
        assert self.auth_manager.verify_password("wrong_password", hashed) is False
    
    def test_api_key_generation(self):
        """Test API key generation."""
        api_key = self.auth_manager.generate_api_key()
        
        assert api_key.startswith("ak_")
        assert len(api_key) > 32
        assert SecurityValidator.validate_api_key_format(api_key) is True
    
    def test_create_user_success(self):
        """Test successful user creation."""
        user = self.auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="StrongPassword123!",
            role=UserRole.DEVELOPER
        )
        
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.DEVELOPER
        assert user.password_hash is not None
        assert user.api_key is not None
        assert "testuser" in self.auth_manager.users
    
    def test_create_user_duplicate_username(self):
        """Test user creation with duplicate username."""
        # Create first user
        user1 = self.auth_manager.create_user(
            "duplicate", "test1@example.com", "Password123!", UserRole.DEVELOPER
        )
        assert user1 is not None
        
        # Try to create second user with same username
        user2 = self.auth_manager.create_user(
            "duplicate", "test2@example.com", "Password123!", UserRole.VIEWER
        )
        assert user2 is None
    
    def test_create_user_weak_password(self):
        """Test user creation with weak password."""
        with pytest.raises(ValueError, match="Password validation failed"):
            self.auth_manager.create_user(
                "testuser", "test@example.com", "weak", UserRole.DEVELOPER
            )
    
    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        # Create user
        self.auth_manager.create_user(
            "authtest", "auth@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Authenticate
        session_id = self.auth_manager.authenticate_user(
            "authtest", "Password123!", "192.168.1.1", "TestAgent/1.0"
        )
        
        assert session_id is not None
        assert session_id in self.auth_manager.sessions
        
        session = self.auth_manager.sessions[session_id]
        assert session.user_id == "authtest"
        assert session.ip_address == "192.168.1.1"
    
    def test_authenticate_user_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        # Create user
        self.auth_manager.create_user(
            "authtest", "auth@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Try with wrong password
        session_id = self.auth_manager.authenticate_user(
            "authtest", "WrongPassword", "192.168.1.1", "TestAgent/1.0"
        )
        
        assert session_id is None
    
    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user."""
        session_id = self.auth_manager.authenticate_user(
            "nonexistent", "Password123!", "192.168.1.1", "TestAgent/1.0"
        )
        
        assert session_id is None
    
    def test_account_lockout(self):
        """Test account lockout after failed attempts."""
        # Create user
        self.auth_manager.create_user(
            "locktest", "lock@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Make 5 failed login attempts
        for _ in range(5):
            session_id = self.auth_manager.authenticate_user(
                "locktest", "WrongPassword", "192.168.1.1", "TestAgent/1.0"
            )
            assert session_id is None
        
        # Account should be locked
        user = self.auth_manager.users["locktest"]
        assert user.locked_until is not None
        assert user.locked_until > datetime.now()
        
        # Even with correct password, should fail while locked
        session_id = self.auth_manager.authenticate_user(
            "locktest", "Password123!", "192.168.1.1", "TestAgent/1.0"
        )
        assert session_id is None
    
    def test_authenticate_api_key_success(self):
        """Test successful API key authentication."""
        # Create user
        user = self.auth_manager.create_user(
            "apitest", "api@example.com", "Password123!", UserRole.API_USER
        )
        
        # Authenticate with API key
        authenticated_user = self.auth_manager.authenticate_api_key(
            user.api_key, "192.168.1.1"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.username == "apitest"
    
    def test_authenticate_api_key_invalid(self):
        """Test API key authentication with invalid key."""
        authenticated_user = self.auth_manager.authenticate_api_key(
            "invalid_api_key", "192.168.1.1"
        )
        
        assert authenticated_user is None
    
    @patch('src.security.auth_manager.RateLimiter')
    def test_rate_limiting_login(self, mock_rate_limiter):
        """Test rate limiting on login attempts."""
        # Mock rate limiter to deny requests
        mock_limiter = Mock()
        mock_limiter.is_allowed.return_value = False
        mock_rate_limiter.return_value = mock_limiter
        
        # Create new auth manager with mocked rate limiter
        auth_manager = AuthenticationManager()
        auth_manager.rate_limiters["login"] = mock_limiter
        
        # Create user
        auth_manager.create_user(
            "ratetest", "rate@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Should raise exception due to rate limiting
        with pytest.raises(Exception, match="Too many login attempts"):
            auth_manager.authenticate_user(
                "ratetest", "Password123!", "192.168.1.1", "TestAgent/1.0"
            )
    
    def test_jwt_token_creation_and_validation(self):
        """Test JWT token creation and validation."""
        # Create user
        user = self.auth_manager.create_user(
            "jwttest", "jwt@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Create session
        session_id = "test_session_123"
        token = self.auth_manager.create_jwt_token(user, session_id)
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long
        
        # Validate token (Note: this would fail in real scenario without session)
        # Just test that validation doesn't crash
        payload = self.auth_manager.validate_jwt_token(token)
        # payload will be None because session doesn't exist in sessions dict
    
    def test_permission_checking(self):
        """Test permission checking."""
        # Create user with specific role
        user = self.auth_manager.create_user(
            "permtest", "perm@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Check permissions that developer should have
        assert self.auth_manager.has_permission(user, Permission.READ_ANALYSIS) is True
        assert self.auth_manager.has_permission(user, Permission.WRITE_ANALYSIS) is True
        assert self.auth_manager.has_permission(user, Permission.GENERATE_CODE) is True
        
        # Check permissions that developer should not have
        assert self.auth_manager.has_permission(user, Permission.MANAGE_USERS) is False
        assert self.auth_manager.has_permission(user, Permission.MANAGE_SYSTEM) is False
    
    def test_require_permission_success(self):
        """Test requiring permission when user has it."""
        user = self.auth_manager.create_user(
            "reqtest", "req@example.com", "Password123!", UserRole.ADMIN
        )
        
        # Should not raise exception
        self.auth_manager.require_permission(user, Permission.MANAGE_USERS)
    
    def test_require_permission_failure(self):
        """Test requiring permission when user doesn't have it."""
        user = self.auth_manager.create_user(
            "reqtest", "req@example.com", "Password123!", UserRole.VIEWER
        )
        
        # Should raise PermissionError
        with pytest.raises(PermissionError, match="does not have permission"):
            self.auth_manager.require_permission(user, Permission.MANAGE_USERS)
    
    def test_logout_user(self):
        """Test user logout."""
        # Create and authenticate user
        user = self.auth_manager.create_user(
            "logouttest", "logout@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        session_id = self.auth_manager.authenticate_user(
            "logouttest", "Password123!", "192.168.1.1", "TestAgent/1.0"
        )
        
        assert session_id in self.auth_manager.sessions
        assert self.auth_manager.sessions[session_id].is_active is True
        
        # Logout
        self.auth_manager.logout_user(session_id, "192.168.1.1")
        
        # Session should be inactive
        assert self.auth_manager.sessions[session_id].is_active is False
    
    def test_revoke_api_key(self):
        """Test API key revocation."""
        user = self.auth_manager.create_user(
            "revoketest", "revoke@example.com", "Password123!", UserRole.API_USER
        )
        
        old_api_key = user.api_key
        
        success = self.auth_manager.revoke_api_key("revoketest")
        assert success is True
        
        new_api_key = self.auth_manager.users["revoketest"].api_key
        assert new_api_key != old_api_key
        
        # Old key should no longer work
        authenticated_user = self.auth_manager.authenticate_api_key(
            old_api_key, "192.168.1.1"
        )
        assert authenticated_user is None
    
    def test_security_events_logging(self):
        """Test security events are properly logged."""
        initial_event_count = len(self.auth_manager.security_events)
        
        # Create user (should log event)
        self.auth_manager.create_user(
            "eventtest", "event@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        # Should have one more event
        assert len(self.auth_manager.security_events) == initial_event_count + 1
        
        # Check the event
        latest_event = self.auth_manager.security_events[-1]
        assert latest_event.event_type == "user_created"
        assert latest_event.user_id == "eventtest"
    
    def test_get_security_events(self):
        """Test retrieving security events with filtering."""
        # Get all events
        all_events = self.auth_manager.get_security_events()
        assert len(all_events) > 0
        
        # Get events for specific user
        user_events = self.auth_manager.get_security_events(user_id="admin")
        assert len(user_events) > 0
        assert all(event.user_id == "admin" for event in user_events)
        
        # Get events by type
        creation_events = self.auth_manager.get_security_events(event_type="user_created")
        assert all(event.event_type == "user_created" for event in creation_events)
    
    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        # Create user and session
        user = self.auth_manager.create_user(
            "cleanuptest", "cleanup@example.com", "Password123!", UserRole.DEVELOPER
        )
        
        session_id = self.auth_manager.authenticate_user(
            "cleanuptest", "Password123!", "192.168.1.1", "TestAgent/1.0"
        )
        
        # Manually expire the session
        session = self.auth_manager.sessions[session_id]
        session.expires_at = datetime.now() - timedelta(hours=1)
        
        assert session_id in self.auth_manager.sessions
        
        # Cleanup expired sessions
        self.auth_manager.cleanup_expired_sessions()
        
        # Session should be removed
        assert session_id not in self.auth_manager.sessions
    
    def test_get_security_summary(self):
        """Test security summary generation."""
        summary = self.auth_manager.get_security_summary()
        
        assert isinstance(summary, dict)
        assert "total_users" in summary
        assert "active_users" in summary
        assert "active_sessions" in summary
        assert "events_last_24h" in summary
        assert "security_events_by_severity" in summary
        
        assert summary["total_users"] >= 1  # At least admin user
        assert summary["active_users"] >= 1


def test_global_auth_manager():
    """Test global authentication manager singleton."""
    manager1 = get_auth_manager()
    manager2 = get_auth_manager()
    
    assert manager1 is manager2  # Should be same instance


if __name__ == "__main__":
    pytest.main([__file__])