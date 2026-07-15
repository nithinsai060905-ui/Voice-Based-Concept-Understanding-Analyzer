import hashlib
import os
import secrets
import json
import base64
import hmac
import time

def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a secure random salt."""
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  # Number of iterations
    )
    return salt.hex() + "$" + key.hex()

def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a password against its PBKDF2-HMAC-SHA256 hash."""
    try:
        salt_hex, key_hex = hashed_password.split("$")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        
        actual_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return hmac.compare_digest(expected_key, actual_key)
    except Exception:
        return False

# Base64URL encoding/decoding helpers
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

def _b64url_decode(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)

def create_jwt_token(payload: dict, secret_key: str, expires_in_seconds: int = 3600 * 24) -> str:
    """
    Creates an HS256 signed JWT token using only Python standard libraries.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    
    # Set expiration
    payload = payload.copy()
    payload["exp"] = int(time.time()) + expires_in_seconds
    
    header_b64 = _b64url_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = _b64url_encode(json.dumps(payload).encode('utf-8'))
    
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    
    signature = hmac.new(
        secret_key.encode('utf-8'),
        signing_input,
        hashlib.sha256
    ).digest()
    
    signature_b64 = _b64url_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{signature_b64}"

def decode_jwt_token(token: str, secret_key: str) -> dict:
    """
    Decodes and verifies an HS256 JWT token.
    Raises ValueError for invalid token, signature, or expired token.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token structure")
            
        header_b64, payload_b64, signature_b64 = parts
        
        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_signature = hmac.new(
            secret_key.encode('utf-8'),
            signing_input,
            hashlib.sha256
        ).digest()
        
        actual_signature = _b64url_decode(signature_b64)
        
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise ValueError("Invalid signature")
            
        # Parse payload
        payload = json.loads(_b64url_decode(payload_b64).decode('utf-8'))
        
        # Verify expiration
        if "exp" in payload and payload["exp"] < time.time():
            raise ValueError("Token has expired")
            
        return payload
    except Exception as e:
        raise ValueError(f"Token validation failed: {str(e)}")
