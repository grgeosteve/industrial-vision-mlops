#!/bin/bash
set -e

# Check .env exists
if [ ! -f .env ]; then
    echo ".env file not found! Please create a .env file with the necessary variables."
    exit 1
fi

# Load variables from .env file
export $(grep -v '^#' .env | xargs)

echo "Initialising DVC..."
dvc init

echo "Configuring DVC remote on S3 bucket..."
dvc remote add origin s3://$DAGSHUB_REPO
dvc remote modify origin endpointurl https://dagshub.com/api/v1/repo-buckets/s3/$DAGSHUB_USERNAME
dvc remote modify origin region us-east-1
dvc config core.remote origin

# Local project telemetry opt-out
dvc config core.analytics false

echo "MLOps setup complete."
