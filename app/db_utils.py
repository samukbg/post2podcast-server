import sqlite3
import os
import logging
import time # Added import for time
from datetime import datetime, timezone
from typing import Optional # Added for type hinting

logger = logging.getLogger(__name__)

# Determine database directory. Default to '/app/data' to match common Fly.io volume mount.
# Allow override via DB_DIR for local testing if needed.
DB_DIR = os.environ.get("DB_DIR", "/app/data") # Changed default from /data to /app/data
DATABASE_URL = os.path.join(DB_DIR, "post2podcast_usage.db")

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(DATABASE_URL), exist_ok=True)
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    """Initializes the database and creates the free_usage table if it doesn't exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS free_usage (
                user_identifier TEXT PRIMARY KEY, -- email:site_url
                credits_used INTEGER DEFAULT 0,
                first_used_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                stripe_subscription_id TEXT PRIMARY KEY,
                stripe_customer_id TEXT,
                wp_user_id TEXT,                  -- WordPress User ID
                wp_user_email TEXT,
                wp_site_url TEXT,
                status TEXT,                      -- e.g., 'active', 'canceled', 'past_due', 'trialing'
                current_period_end INTEGER,       -- Unix timestamp (INTEGER for SQLite)
                cancel_at_period_end BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_wp_user_id ON subscriptions(wp_user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_wp_user_email ON subscriptions(wp_user_email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id ON subscriptions(stripe_customer_id);")
        conn.commit()
        logger.info(f"Database initialized at {DATABASE_URL}. Tables free_usage and subscriptions ensured.")
    except Exception as e:
        logger.error(f"Error initializing database at {DATABASE_URL}: {e}")
    finally:
        if conn:
            conn.close()

def get_free_credits_used(user_identifier: str) -> int:
    """
    Queries the database for the given user_identifier and returns credits_used.
    Returns 0 if the user_identifier is not found.
    """
    conn = get_db_connection()
    credits = 0
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT credits_used FROM free_usage WHERE user_identifier = ?", (user_identifier,))
        row = cursor.fetchone()
        if row:
            credits = row['credits_used']
    except Exception as e:
        logger.error(f"Error getting free credits for {user_identifier}: {e}")
        # Depending on policy, you might want to raise the error or return a value indicating error
    finally:
        if conn:
            conn.close()
    return credits

def increment_free_credit_usage(user_identifier: str):
    """
    Increments the credits_used for the given user_identifier.
    If the user_identifier does not exist, it creates a new record with credits_used = 1.
    Updates last_used_timestamp.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Use UTC for timestamps
        current_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # UPSERT behavior: Insert or update on conflict
        cursor.execute("""
            INSERT INTO free_usage (user_identifier, credits_used, last_used_timestamp, first_used_timestamp)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(user_identifier) DO UPDATE SET
                credits_used = credits_used + 1,
                last_used_timestamp = excluded.last_used_timestamp
        """, (user_identifier, current_time_utc, current_time_utc)) # first_used_timestamp only on insert
        
        # For SQLite versions older than 3.24.0 that don't support ON CONFLICT with excluded,
        # you might need a two-step select then insert/update.
        # However, modern SQLite should support this.

        conn.commit()
        logger.info(f"Incremented/recorded free credit usage for {user_identifier}")
    except Exception as e:
        logger.error(f"Error incrementing free credits for {user_identifier}: {e}")
    finally:
        if conn:
            conn.close()

# Placeholder for a more advanced subscription check if you build a user/subscription table
# def is_user_subscribed(user_identifier: str) -> bool:
#     """Checks if the user associated with user_identifier has an active subscription."""
#     # This would query your own subscriptions table, populated by Stripe webhooks.
#     # For now, this is a placeholder. The main logic will rely on whether an API key is passed.
#     logger.debug(f"Placeholder: is_user_subscribed check for {user_identifier}")
#     return False

def add_or_update_subscription(
    stripe_subscription_id: str,
    stripe_customer_id: str,
    wp_user_id: str,
    wp_user_email: str,
    wp_site_url: str,
    status: str,
    current_period_end: int, # Unix timestamp
    cancel_at_period_end: bool
):
    """Adds a new subscription or updates an existing one."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        current_time_utc_iso = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            INSERT INTO subscriptions (
                stripe_subscription_id, stripe_customer_id, wp_user_id, wp_user_email, wp_site_url, 
                status, current_period_end, cancel_at_period_end, updated_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stripe_subscription_id) DO UPDATE SET
                stripe_customer_id = excluded.stripe_customer_id,
                wp_user_id = excluded.wp_user_id,
                wp_user_email = excluded.wp_user_email,
                wp_site_url = excluded.wp_site_url,
                status = excluded.status,
                current_period_end = excluded.current_period_end,
                cancel_at_period_end = excluded.cancel_at_period_end,
                updated_at = excluded.updated_at
        """, (
            stripe_subscription_id, stripe_customer_id, wp_user_id, wp_user_email, wp_site_url,
            status, current_period_end, 1 if cancel_at_period_end else 0, 
            current_time_utc_iso, current_time_utc_iso # created_at will only be set on INSERT
        ))
        conn.commit()
        logger.info(f"Subscription {stripe_subscription_id} added/updated for user {wp_user_id} at {wp_site_url} with status {status}.")
    except Exception as e:
        logger.error(f"Error adding/updating subscription {stripe_subscription_id}: {e}")
    finally:
        if conn:
            conn.close()

def get_subscription_status_by_identifier(user_identifier: str) -> Optional[dict]:
    """
    Retrieves active subscription status for a user based on 'email:site_url' identifier.
    Returns a dict with 'status' and 'current_period_end' or None if no active subscription.
    """
    conn = get_db_connection()
    try:
        if not isinstance(user_identifier, str):
            logger.error(f"Invalid user_identifier type for subscription status check: {type(user_identifier)}")
            return None
        email, site_url = user_identifier.split(':', 1)
        cursor = conn.cursor()
        # Check for an active subscription that hasn't ended
        # Using current_time directly in SQL might be tricky with timezones in SQLite,
        # so comparing current_period_end with current time in Python is safer if needed.
        # For simplicity, we'll just check status = 'active' here.
        # A more robust check would also ensure current_period_end > time.time().
        cursor.execute("""
            SELECT status, current_period_end 
            FROM subscriptions 
            WHERE wp_user_email = ? AND wp_site_url = ? AND status = 'active' 
            ORDER BY updated_at DESC LIMIT 1 
        """, (email, site_url)) # Assumes email and site_url are stored consistently
        row = cursor.fetchone()
        if row:
            # Check if period is still valid
            if row['current_period_end'] and int(row['current_period_end']) > time.time():
                return {"status": row['status'], "current_period_end": row['current_period_end']}
            else:
                logger.info(f"Active subscription found for {user_identifier} but period has ended.")
                return None # Period ended
        return None
    except ValueError:
        logger.error(f"Invalid user_identifier format for subscription status check: {user_identifier}")
        return None
    except Exception as e:
        logger.error(f"Error getting subscription status for {user_identifier}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_subscription_status_by_id(stripe_subscription_id: str, new_status: str, current_period_end: Optional[int] = None, cancel_at_period_end: Optional[bool] = None):
    """Updates the status and optionally current_period_end of a specific subscription."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        current_time_utc_iso = datetime.now(timezone.utc).isoformat()
        
        fields_to_update = ["status = ?", "updated_at = ?"]
        params = [new_status, current_time_utc_iso]
        
        if current_period_end is not None:
            fields_to_update.append("current_period_end = ?")
            params.append(current_period_end)
        
        if cancel_at_period_end is not None:
            fields_to_update.append("cancel_at_period_end = ?")
            params.append(1 if cancel_at_period_end else 0)
            
        params.append(stripe_subscription_id) # For the WHERE clause
        
        sql = f"UPDATE subscriptions SET {', '.join(fields_to_update)} WHERE stripe_subscription_id = ?"
        
        cursor.execute(sql, tuple(params))
        conn.commit()
        
        if cursor.rowcount > 0:
            logger.info(f"Updated status for subscription {stripe_subscription_id} to {new_status}.")
        else:
            logger.warning(f"Attempted to update status for non-existent subscription {stripe_subscription_id}.")
            # Optionally, call add_or_update_subscription if it should create if not exists,
            # but this function is meant for targeted status updates.
            
    except Exception as e:
        logger.error(f"Error updating subscription status for {stripe_subscription_id}: {e}")
    finally:
        if conn:
            conn.close()
