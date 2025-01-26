from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Importowanie modeli i konfiguracji bazy danych
from models.database import Base
from models.project_model import Project  # Importuj wszystkie modele, które będą migrowane

# Konfiguracja Alembic
config = context.config

# Ustawienie logowania
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Przekazanie metadanych modeli do Alembic (dla autogeneracji)
target_metadata = Base.metadata

# Ustawienie ścieżki do bazy danych w pliku alembic.ini
config.set_main_option("sqlalchemy.url", "sqlite:///./projects.db")


def run_migrations_offline() -> None:
    """Uruchom migracje w trybie offline."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Uruchom migracje w trybie online."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
