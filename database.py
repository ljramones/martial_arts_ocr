"""
Database configuration and operations for Martial Arts OCR.
Handles SQLite database setup, sessions, and utilities.
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
import logging

from config import get_config

logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create engine with SQLite-specific settings
engine_kwargs = {
    'echo': config.DEBUG,  # Log SQL queries in debug mode
    'poolclass': StaticPool,
    'pool_pre_ping': True,
    'connect_args': {
        'check_same_thread': False,  # Allow multiple threads
        'timeout': 30,  # Connection timeout
    }
}

# Create the engine
engine = create_engine(config.DATABASE_URL, **engine_kwargs)


# Enable foreign key support for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints and other SQLite optimizations."""
    cursor = dbapi_connection.cursor()
    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys=ON")
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")
    # Optimize for faster operations
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=10000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


# Create the session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)
# Create a scoped session for thread safety
Session = scoped_session(SessionLocal)

# Create a declarative base for models
Base = declarative_base()


def init_db():
    """Initialize the database, creating all tables."""
    try:
        # Import all models to ensure they're registered
        from models import Document, Page, ProcessingResult

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")

        # Create necessary directories
        from pathlib import Path
        db_dir = Path(config.DATABASE_URL.replace('sqlite:///', '')).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def drop_all_tables():
    """Drop all tables. Use with caution!"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


def reset_database():
    """Reset the database by dropping and recreating all tables."""
    try:
        logger.warning("Resetting database - all data will be lost!")
        drop_all_tables()
        init_db()
        logger.info("Database reset completed")
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.
    Automatically handles session lifecycle and error rollback.

    Usage:
        with get_db_session() as session:
            user = session.query(User).first()
            # session is automatically committed and closed
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_session():
    """Get a new database session. Remember to close it when done!"""
    return SessionLocal()


class DatabaseManager:
    """Database management utilities."""

    @staticmethod
    def check_connection():
        """Check if database connection is working."""
        try:
            with get_db_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    @staticmethod
    def get_table_info():
        """Get information about database tables."""
        try:
            with get_db_session() as session:
                # Get table names
                result = session.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in result.fetchall()]

                table_info = {}
                for table in tables:
                    # Get row count for each table
                    count_result = session.execute(f"SELECT COUNT(*) FROM {table}")
                    count = count_result.fetchone()[0]
                    table_info[table] = count

                return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return {}

    @staticmethod
    def vacuum_database():
        """Optimize database by running VACUUM."""
        try:
            with engine.connect() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False

    @staticmethod
    def backup_database(backup_path: str):
        """Create a backup of the database."""
        try:
            import shutil
            from pathlib import Path

            # Get source database path
            db_path = config.DATABASE_URL.replace('sqlite:///', '')

            # Create backup
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_path, backup_file)

            logger.info(f"Database backed up to: {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

    @staticmethod
    def get_database_size():
        """Get database file size in bytes."""
        try:
            db_path = config.DATABASE_URL.replace('sqlite:///', '')
            if os.path.exists(db_path):
                return os.path.getsize(db_path)
            return 0
        except Exception as e:
            logger.error(f"Failed to get database size: {e}")
            return 0


# Database utility functions
def count_documents():
    """Count total number of documents."""
    try:
        with get_db_session() as session:
            from models import Document
            return session.query(Document).count()
    except Exception as e:
        logger.error(f"Failed to count documents: {e}")
        return 0


def count_processed_documents():
    """Count number of successfully processed documents."""
    try:
        with get_db_session() as session:
            from models import Document
            return session.query(Document).filter_by(status='completed').count()
    except Exception as e:
        logger.error(f"Failed to count processed documents: {e}")
        return 0


def get_recent_documents(limit: int = 10):
    """Get most recently uploaded documents."""
    try:
        with get_db_session() as session:
            from models import Document
            return session.query(Document) \
                .order_by(Document.upload_date.desc()) \
                .limit(limit) \
                .all()
    except Exception as e:
        logger.error(f"Failed to get recent documents: {e}")
        return []


def cleanup_failed_uploads():
    """Remove database records for failed uploads with missing files."""
    try:
        with get_db_session() as session:
            from models import Document
            from config import get_upload_path

            # Get all documents
            documents = session.query(Document).all()
            removed_count = 0

            for doc in documents:
                file_path = get_upload_path(doc.filename)
                if not file_path.exists():
                    # File is missing, remove database record
                    session.delete(doc)
                    removed_count += 1
                    logger.info(f"Removed record for missing file: {doc.filename}")

            session.commit()
            logger.info(f"Cleanup completed. Removed {removed_count} orphaned records.")
            return removed_count
    except Exception as e:
        logger.error(f"Failed to cleanup failed uploads: {e}")
        return 0


def get_database_stats():
    """Get comprehensive database statistics."""
    try:
        stats = {
            'connection_ok': DatabaseManager.check_connection(),
            'database_size': DatabaseManager.get_database_size(),
            'table_info': DatabaseManager.get_table_info(),
            'total_documents': count_documents(),
            'processed_documents': count_processed_documents(),
        }

        # Calculate processing success rate
        if stats['total_documents'] > 0:
            stats['success_rate'] = (stats['processed_documents'] / stats['total_documents']) * 100
        else:
            stats['success_rate'] = 0

        return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


# Health check function
def health_check():
    """Perform database health check."""
    try:
        health = {
            'status': 'healthy',
            'connection': DatabaseManager.check_connection(),
            'tables_exist': len(DatabaseManager.get_table_info()) > 0,
            'total_documents': count_documents(),
        }

        if not health['connection']:
            health['status'] = 'unhealthy'
            health['error'] = 'Database connection failed'
        elif not health['tables_exist']:
            health['status'] = 'unhealthy'
            health['error'] = 'Database tables not found'

        return health
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


# CLI utility functions for database management
def main():
    """Command-line interface for database operations."""
    import argparse

    parser = argparse.ArgumentParser(description='Database management utilities')
    parser.add_argument('command', choices=[
        'init', 'reset', 'stats', 'backup', 'vacuum', 'cleanup', 'health'
    ], help='Command to execute')
    parser.add_argument('--backup-path', help='Path for database backup')

    args = parser.parse_args()

    if args.command == 'init':
        init_db()
        print("Database initialized successfully")

    elif args.command == 'reset':
        confirm = input("This will delete all data. Are you sure? (yes/no): ")
        if confirm.lower() == 'yes':
            reset_database()
            print("Database reset completed")
        else:
            print("Operation cancelled")

    elif args.command == 'stats':
        stats = get_database_stats()
        print("\nDatabase Statistics:")
        print(f"Connection: {'OK' if stats['connection_ok'] else 'FAILED'}")
        print(f"Size: {stats['database_size']} bytes")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Processed Documents: {stats['processed_documents']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print("\nTables:")
        for table, count in stats['table_info'].items():
            print(f"  {table}: {count} rows")

    elif args.command == 'backup':
        if not args.backup_path:
            print("Error: --backup-path required for backup command")
            return
        if DatabaseManager.backup_database(args.backup_path):
            print(f"Database backed up to: {args.backup_path}")
        else:
            print("Backup failed")

    elif args.command == 'vacuum':
        if DatabaseManager.vacuum_database():
            print("Database vacuumed successfully")
        else:
            print("Vacuum failed")

    elif args.command == 'cleanup':
        removed = cleanup_failed_uploads()
        print(f"Cleanup completed. Removed {removed} orphaned records.")

    elif args.command == 'health':
        health = health_check()
        print(f"\nDatabase Health: {health['status'].upper()}")
        if health['status'] == 'unhealthy':
            print(f"Error: {health.get('error', 'Unknown error')}")
        else:
            print(f"Connection: {'OK' if health['connection'] else 'FAILED'}")
            print(f"Tables: {'Present' if health['tables_exist'] else 'Missing'}")
            print(f"Documents: {health['total_documents']}")


if __name__ == '__main__':
    main()