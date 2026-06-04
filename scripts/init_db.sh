#!/bin/bash
set -e

POSTGRES_USER="${POSTGRES_USER}"
DB_NAME="${DB_NAME}"
DB_USER="${DB_USER}"
DB_PASS="${DB_PASSWORD}"

echo "Starting database initialization..."

# STEP 1: Connect to 'postgres' to create the user and database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    -- Create the application user if it doesn't exist
    DO \$$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER') THEN
            CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';
        END IF;
    END
    \$$;

    -- Create the database
    SELECT 'CREATE DATABASE $DB_NAME'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec

    -- Grant database-level privileges
    GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOSQL

# STEP 2: Connect directly to the new database to alter its public schema
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$DB_NAME" <<-EOSQL
    -- Ensure the user owns or can write to the public schema of THIS database
    ALTER SCHEMA public OWNER TO $DB_USER;

    -- Alternative if you want to keep 'postgres' as owner but grant full access:
    -- GRANT ALL ON SCHEMA public TO $DB_USER;
EOSQL

echo "Database '$DB_NAME' and user '$DB_USER' are successfully configured."