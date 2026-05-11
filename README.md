# Vision MLOps for Industrial Datasets

An end-to-end MLOps pipeline applied to automated object detection in logistics environments (LOCO dataset).

While real-time industrial computer vision is typically deployed on the Edge for low latency, **this project explicitly focuses on a Cloud-Native Microservices architecture.** It simulates a centralised, asynchronous inference API designed for batch-processing warehouse footage, facility dashboarding, and digital-twin integration. 

This project demonstrates data engineering, strict dependency isolation, reproducible experiment tracking, and microservices-based cloud deployment of computer vision models. The project is still under development.

## Technologies 
- **Environment:** Conda (Local) / Docker (RunPod)
- **Data:** DagsHub S3
- **Data Versioning:** DVC -> DagsHub S3
- **Experiment Tracking:** MLflow -> DagsHub
- **Machine Learning Framework:** PyTorch

## Architecture
This project separates the ML lifecycle into strictly versioned phases:

1. **Data Ingestion:** Automated fetching and validating of raw datasets.
2. **Data Processing:** (In Progress) Conversion of COCO dataset to normalised YOLO format and EDA
3. **Model Training:** (Planned)
4. **Deployment:** (Planned)

## Repository Structure

```text
├── configs/                 # YAML configuration files
│   └── data_config.yaml     # Dataset ingestion and processing settings
├── data/
│   ├── external/            # Data from third party sources (Git-ignored)
│   ├── processed/           # Normalised YOLO format data (Git-ignored)
│   └── raw/                 # Original COCO datasets (Git-ignored)
├── init_mlops.sh            # Environment and tracking initialisation script
├── LICENSE
├── models/                  # Trained models, ONNX exports, and model summaries
├── notebooks/               # Sandboxed Exploratory Data Analysis (EDA)
├── pyproject.toml           # Project metadata and build configuration
├── README.md                # Project documentation
├── requirements-base-dev.txt# Testing, typing, and EDA dependencies
├── requirements-base.txt    # Core dependencies for data engineering pipeline
├── src/                     # Production-ready pipeline source code
│   ├── api/                 # Microservice deployment code (FastAPI)
│   ├── data/                # Data ingestion and processing orchestrators
│   │   └── ingest_data.py
│   ├── models/              # Model training and evaluation scripts
│   ├── parsers/             # Agnostic conversion logic (e.g., COCO to YOLO)
│   ├── paths.py             # Global path definitions
│   └── utils/               # Shared helper functions
│       └── file_ops.py
└── tests/                   # Pytest suite
    ├── integration/         # Pipeline and API integration tests
    └── unit/                # Utility and parser unit tests
```

## Stage 1: Environment Setup 
1. **Clone the repository:**

    ```bash
    git clone https://github.com/grgeosteve/industrial-vision-mlops.git
    cd ./industrial-vision-mlops
    ```
2. **Install dependencies:**

    ```bash
    # Create and activate fresh conda environment
    conda create -n industrial-vision python=3.11 -y
    conda activate industrial-vision 

    # Install the core pipeline dependencies
    pip install -r requirements-base.txt

    # Install developer tools (testing, EDA, etc.)
    pip install -r requirements-base-dev.txt

    # Install the project locally in editable mode
    pip install -e .
    ```

3. **Create the secrets file:**
    Create a `.env` file in the root directory

    ```text
    DAGSHUB_USERNAME=your_username
    DAGSHUB_REPO=your_repo_name
    AWS_ACCESS_KEY_ID=your_aws_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret
    ```

    **Note:** This pipeline currently utilises DagsHub's S3-compatible storage backend.
    Environment variables (AWS_ACCESS_KEY_ID, etc.) are used to ensure compatibility with standard S3 libraries (like `boto3`) and to demonstrate an enterprise-compliant architecture. However, execution on native AWS infrastructure has not been explicitly tested.

4. **Initialise the architecture:**

    ```bash
    chmod +x init_mlops.sh
    ./init_mlops.sh
    ```

## Stage 2: Data Acquisition & Processing
### Data Retrieval (Default)
Because the raw data is version-controlled as an immutable artifact, you do not need to download it manually. Once your `.env` and DagsHub remote are configured via `init_mlops.sh`, simply pull the data:

```bash
    dvc pull
```

### Manual Data Ingestion (Admin Only)
*If you are updating the dataset or rebuilding the DVC pipeline from scratch, you can trigger the raw ingestion script. This downloads the LOCO dataset directly from the TUM servers.*

*Credits to the TUM team for creating and supplying the dataset in their repository:* https://github.com/tum-fml/loco

>LOCO: Logistics Objects in Context    
>Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J.     
>IEEE International Conference on Machine Learning and Applications (ICMLA) 2020

```bash
    python src/data/ingest_data.py --dataset loco --config configs/data_config.yaml
```

### Exploratory Data Analysis (EDA)
All exploratory notebooks are isolated in the `notebooks/` directory to prevent environment pollution. To run the EDA notebooks, ensure you have installed the development dependencies using `requirements-base-dev.txt`.

From within the conda environment launch `marimo`:

```bash
    marimo edit notebooks/01_EDA_raw_loco.py
```

## Testing
Unit and pipeline integration tests have been built to verify the correct execution of the utility functions, API integration, and processing logic.

Additionally, `mypy` is used for ensuring typing adherence.

```bash
# Verify strict typing
mypy src/ tests/

# Perform unit and integration tests
pytest tests/ -v
```