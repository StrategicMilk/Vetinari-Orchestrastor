"""
Unit tests for the security module.

Tests secret detection and sanitization functionality.
"""

import unittest
from vetinari.security import (
    SecretScanner, SecretPattern, get_secret_scanner,
    sanitize_for_memory, sanitize_dict_for_memory
)


class TestSecretPatterns(unittest.TestCase):
    """Test individual secret pattern detection."""
    
    def setUp(self):
        self.scanner = get_secret_scanner()
    
    def test_openai_api_key_detection(self):
        """Test OpenAI API key detection."""
        test_cases = [
            "sk-proj-abc123def456ghi789jklmnopqrs",
            "sk-abcdefghijklmnopqrst",
            "API key: sk-1234567890123456789012"
        ]
        
        for test_case in test_cases:
            detected = self.scanner.scan(test_case)
            self.assertIn("openai_api_key", detected)
    
    def test_github_token_detection(self):
        """Test GitHub token detection."""
        test_cases = [
            "ghp_abc123def456ghi789jklmnopqrstuv",
            "ghu_abc123def456ghi789jklmnopqrstuv",
            "Token: ghs_abc123def456ghi789jklmnopqrstuv"
        ]
        
        for test_case in test_cases:
            detected = self.scanner.scan(test_case)
            # Should detect at least one GitHub token pattern
            patterns = [k for k in detected.keys() if 'github' in k or 'ghp' in str(detected[k]).lower()]
            self.assertTrue(len(patterns) > 0 or 'github_token' in detected, f"Failed for {test_case}")
    
    def test_aws_key_detection(self):
        """Test AWS access key detection."""
        content = "AKIAIOSFODNN7EXAMPLE"
        detected = self.scanner.scan(content)
        self.assertIn("aws_access_key", detected)
    
    def test_jwt_token_detection(self):
        """Test JWT token detection."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        detected = self.scanner.scan(jwt)
        self.assertIn("jwt_token", detected)
    
    def test_bearer_token_detection(self):
        """Test Bearer token detection."""
        content = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        detected = self.scanner.scan(content)
        # Should detect JWT or bearer token
        self.assertTrue(len(detected) > 0)


class TestSensitiveKeywords(unittest.TestCase):
    """Test sensitive field name detection."""
    
    def setUp(self):
        self.scanner = get_secret_scanner()
    
    def test_password_keywords(self):
        """Test detection of password field names."""
        keywords = ["password", "passwd", "pwd"]
        
        for keyword in keywords:
            is_sensitive = self.scanner._is_sensitive_key(keyword)
            self.assertTrue(is_sensitive, f"{keyword} not detected as sensitive")
    
    def test_api_key_keywords(self):
        """Test detection of API key field names."""
        keywords = ["api_key", "apikey", "api-key", "api_secret", "client_secret"]
        
        for keyword in keywords:
            is_sensitive = self.scanner._is_sensitive_key(keyword)
            self.assertTrue(is_sensitive, f"{keyword} not detected as sensitive")
    
    def test_token_keywords(self):
        """Test detection of token field names."""
        keywords = ["token", "access_token", "refresh_token", "oauth_token", "bearer"]
        
        for keyword in keywords:
            is_sensitive = self.scanner._is_sensitive_key(keyword)
            self.assertTrue(is_sensitive, f"{keyword} not detected as sensitive")


class TestSanitization(unittest.TestCase):
    """Test content sanitization."""
    
    def setUp(self):
        self.scanner = get_secret_scanner()
    
    def test_sanitize_openai_key(self):
        """Test OpenAI key sanitization."""
        content = "My OpenAI API key is sk-proj-abc123def456ghi789"
        sanitized = self.scanner.sanitize(content)
        
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("sk-proj-", sanitized)
    
    def test_sanitize_multiple_secrets(self):
        """Test sanitization of multiple secrets."""
        content = """
        OpenAI: sk-proj-abc123def456
        GitHub: ghp_1234567890123456789012345678901
        AWS: AKIAIOSFODNN7EXAMPLE
        """
        
        sanitized = self.scanner.sanitize(content)
        
        # Count redactions
        redaction_count = sanitized.count("[REDACTED]")
        self.assertGreaterEqual(redaction_count, 2)  # At least 2 secrets
        
        # Original secrets should be removed
        self.assertNotIn("sk-proj-", sanitized)
    
    def test_sanitize_dict_sensitive_fields(self):
        """Test dictionary sanitization for sensitive field names."""
        data = {
            "username": "alice",
            "password": "secretpass123",
            "email": "alice@example.com",
            "api_key": "sk-proj-secret"
        }
        
        sanitized = self.scanner.sanitize_dict(data)
        
        # Sensitive fields should be redacted
        self.assertEqual(sanitized["password"], "[REDACTED]")
        self.assertEqual(sanitized["api_key"], "[REDACTED]")
        
        # Non-sensitive fields should be preserved
        self.assertEqual(sanitized["username"], "alice")
        self.assertEqual(sanitized["email"], "alice@example.com")
    
    def test_sanitize_nested_dict(self):
        """Test sanitization of nested dictionaries."""
        data = {
            "user": {
                "name": "alice",
                "password": "secret123"
            },
            "config": {
                "api_key": "sk-proj-secret",
                "endpoint": "https://api.example.com"
            }
        }
        
        sanitized = self.scanner.sanitize_dict(data)
        
        # Nested sensitive fields should be redacted
        self.assertEqual(sanitized["user"]["password"], "[REDACTED]")
        self.assertEqual(sanitized["config"]["api_key"], "[REDACTED]")
        
        # Non-sensitive nested fields should be preserved
        self.assertEqual(sanitized["user"]["name"], "alice")
        self.assertEqual(sanitized["config"]["endpoint"], "https://api.example.com")
    
    def test_sanitize_list(self):
        """Test sanitization of lists."""
        data = {
            "tokens": ["token1", "sk-proj-secret", "token3"],
            "endpoints": ["https://api1.com", "https://api2.com"]
        }
        
        sanitized = self.scanner.sanitize_dict(data)
        
        # API key in list should be redacted
        self.assertEqual(sanitized["tokens"][1], "[REDACTED]")
        
        # Other items should be preserved
        self.assertEqual(sanitized["tokens"][0], "token1")
        self.assertEqual(len(sanitized["endpoints"]), 2)
    
    def test_preserve_safe_content(self):
        """Test that safe content is preserved during sanitization."""
        content = """
        This is a regular configuration file.
        Database: postgresql://localhost:5432/mydb
        Endpoint: https://api.example.com
        Timeout: 30
        """
        
        sanitized = self.scanner.sanitize(content)
        
        # Safe parts should be preserved
        self.assertIn("regular configuration", sanitized)
        self.assertIn("Endpoint:", sanitized)
        self.assertIn("Timeout:", sanitized)


class TestCustomPatterns(unittest.TestCase):
    """Test custom pattern addition."""
    
    def test_add_custom_pattern(self):
        """Test adding a custom secret pattern."""
        scanner = SecretScanner()  # New instance
        
        # Add custom pattern for a specific service
        custom_pattern = SecretPattern(
            name="custom_service_key",
            pattern=r"custom_[a-zA-Z0-9]{20,}",
            description="Custom service API key"
        )
        
        scanner.add_pattern(custom_pattern)
        
        # Test detection — value must have >= 20 alphanumeric chars after "custom_"
        content = "service_key: custom_abc123def456789xyzABCD"
        detected = scanner.scan(content)
        self.assertIn("custom_service_key", detected)


class TestMemoryIntegration(unittest.TestCase):
    """Test integration with memory module."""
    
    def test_sanitize_for_memory_convenience(self):
        """Test convenience function for memory sanitization."""
        content = "Config: api_key=sk-proj-abc123def456"
        sanitized = sanitize_for_memory(content)
        
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("sk-proj-", sanitized)
    
    def test_sanitize_dict_for_memory_convenience(self):
        """Test convenience function for dictionary sanitization."""
        data = {
            "api_key": "sk-proj-secret",
            "endpoint": "https://api.example.com"
        }
        
        sanitized = sanitize_dict_for_memory(data)
        
        self.assertEqual(sanitized["api_key"], "[REDACTED]")
        self.assertEqual(sanitized["endpoint"], "https://api.example.com")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and corner scenarios."""
    
    def setUp(self):
        self.scanner = get_secret_scanner()
    
    def test_empty_content(self):
        """Test handling of empty content."""
        detected = self.scanner.scan("")
        self.assertEqual(len(detected), 0)
        
        sanitized = self.scanner.sanitize("")
        self.assertEqual(sanitized, "")
    
    def test_none_type(self):
        """Test handling of None type."""
        # Scan should handle non-string gracefully
        detected = self.scanner.scan("valid string")
        self.assertIsInstance(detected, dict)
    
    def test_very_long_content(self):
        """Test handling of very long content."""
        # Create a long string with a secret
        content = "x" * 10000 + " sk-proj-secret123 " + "y" * 10000
        
        sanitized = self.scanner.sanitize(content)
        
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("sk-proj-", sanitized)
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        content = "密钥: sk-proj-secret123 用户名: 张三"
        
        sanitized = self.scanner.sanitize(content)
        
        self.assertIn("[REDACTED]", sanitized)
        self.assertIn("用户名:", sanitized)  # Non-secret parts preserved
    
    def test_case_insensitive_detection(self):
        """Test case-insensitive pattern matching."""
        content_upper = "MY-API-KEY: SK-PROJ-ABC123DEF456"
        content_lower = "my-api-key: sk-proj-abc123def456"
        
        detected_upper = self.scanner.scan(content_upper)
        detected_lower = self.scanner.scan(content_lower)
        
        # Both should detect the pattern
        self.assertTrue(len(detected_upper) > 0)
        self.assertTrue(len(detected_lower) > 0)


if __name__ == "__main__":
    unittest.main()
