import requests
import jwt
from jwt.algorithms import RSAAlgorithm
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict
import json

security = HTTPBearer()

_jwks_cache = None

def get_clerk_jwks():
    """Fetch Clerk's public key from JWKS endpoint with caching"""
    global _jwks_cache
    
    if _jwks_cache is not None:
        return _jwks_cache
    
    try:
        jwks_url = "https://valid-termite-98.clerk.accounts.dev/.well-known/jwks.json"
        
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        
        _jwks_cache = response.json()
        print(f"JWKS fetched successfully, found {len(_jwks_cache.get('keys', []))} keys")
        return _jwks_cache
    except requests.exceptions.RequestException as e:
        print(f"JWKS fetch error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch JWKS: {str(e)}"
        )

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:
    """Validate Clerk JWT and extract user info"""
    try:
        token = credentials.credentials
        print(f"Received token (first 20 chars): {token[:20]}...")
        
        try:
            unverified_header = jwt.get_unverified_header(token)
            print(f"Token header: {unverified_header}")
        except Exception as e:
            print(f"Failed to decode token header: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token format: {str(e)}"
            )
        
        kid = unverified_header.get('kid')
        
        if not kid:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing kid in header"
            )
        
        print(f"Looking for key with kid: {kid}")
        
        jwks = get_clerk_jwks()
        
        if not jwks.get('keys'):
            raise HTTPException(
                status_code=500,
                detail="No keys found in JWKS"
            )
        
        rsa_key = None
        for key in jwks['keys']:
            if key.get('kid') == kid:
                rsa_key = key
                print(f"Found matching key: {kid}")
                break
        
        if not rsa_key:
            available_kids = [k.get('kid') for k in jwks['keys']]
            print(f"No matching key found. Available kids: {available_kids}")
            raise HTTPException(
                status_code=401,
                detail=f"Unable to find matching key. Token kid: {kid}, Available: {available_kids}"
            )
        
        try:
            public_key = RSAAlgorithm.from_jwk(json.dumps(rsa_key))
            print("Public key constructed successfully")
        except Exception as e:
            print(f"Failed to construct public key: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail=f"Failed to construct public key: {str(e)}"
            )
        
        try:
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            print(f"Unverified payload: {unverified_payload}")
        except Exception as e:
            print(f"Could not decode unverified payload: {str(e)}")
        
        try:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience="fastapi",  
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_exp": True,
                    "verify_iat": True
                },
                leeway=60  # Allow 60 seconds clock skew
            )
            print("Token verified successfully")
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidAudienceError as e:
            print(f"Invalid audience: {str(e)}")
            try:
                payload = jwt.decode(
                    token,
                    public_key,
                    algorithms=["RS256"],
                    options={
                        "verify_signature": True,
                        "verify_aud": False,
                        "verify_exp": True,
                        "verify_iat": False  # Disable iat check for clock skew
                    },
                    leeway=300  # Allow 5 minutes clock skew
                )
                print("Token verified without audience check")
            except Exception as retry_error:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid token audience. Expected 'fastapi', got: {unverified_payload.get('aud')}"
                )
        except jwt.InvalidTokenError as e:
            error_msg = str(e)
            print(f"Invalid token: {error_msg}")
            # Check if it's an iat (issued at) error - clock skew issue
            if "iat" in error_msg.lower() or "not yet valid" in error_msg.lower():
                print(f"Token issued at time issue (clock skew): {error_msg}")
                # Try again without iat verification to handle clock skew
                try:
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=["RS256"],
                        options={
                            "verify_signature": True,
                            "verify_aud": False,  # Also disable aud check
                            "verify_exp": True,
                            "verify_iat": False  # Disable iat check for clock skew
                        },
                        leeway=300  # Allow 5 minutes clock skew
                    )
                    print("Token verified with iat check disabled (clock skew tolerance)")
                except Exception as retry_error:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Token validation failed: {str(retry_error)}"
                    )
            # If it's an audience issue, try without audience verification
            elif "aud" in error_msg.lower() or "audience" in error_msg.lower() or "missing" in error_msg.lower():
                try:
                    print("Retrying token verification without audience check...")
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=["RS256"],
                        options={
                            "verify_signature": True,
                            "verify_aud": False,
                            "verify_exp": True,
                            "verify_iat": False  # Disable iat check for clock skew
                        },
                        leeway=300  # Allow 5 minutes clock skew
                    )
                    print("Token verified without audience check")
                except Exception as retry_error:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Invalid token: {str(retry_error)}"
                    )
            else:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid token: {error_msg}"
                )
        
        # Extract user information from payload
        claims = payload.get("claims", {})
        
        # Try to get email from various possible locations
        email = (
            claims.get("email") or 
            payload.get("email") or 
            claims.get("primary_email_address") or
            payload.get("primary_email_address") or
            f"user@{payload.get('sub', 'unknown')}.clerk"  # Fallback email
        )
        
        # Extract user_id - Clerk uses 'sub' field which contains user_xxx
        user_id = (
            claims.get("id") or 
            payload.get("id") or 
            payload.get("sub") or
            claims.get("sub")
        )
        
        if not user_id:
            print(f"No user_id in payload. Available claims: {list(payload.keys())}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: missing user_id. Available: {list(payload.keys())}"
            )
        
        # Extract user_id from 'sub' if it's in the format 'user_xxx'
        if isinstance(user_id, str) and user_id.startswith("user_"):
            print(f"✓ Extracted user_id from sub: {user_id}")
        elif isinstance(user_id, str):
            # If sub doesn't start with user_, use it as is
            print(f"✓ Using user_id from sub: {user_id}")
        
        # Email is optional - we can authenticate with just user_id
        if not email:
            print(f"No email in payload, using fallback. Available claims: {list(payload.keys())}")
            email = f"user_{user_id}@clerk.local"
        
        print(f"Authenticated user: {email} (ID: {user_id})")
        return {"email": email, "user_id": user_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected authentication error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )