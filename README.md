# Vision MLOps for Industrial Datasets

![gates](https://github.com/grgeosteve/industrial-vision-mlops/actions/workflows/gates.yml/badge.svg)

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

1. **Data Ingestion:** Automated fetching and secure extraction of raw datasets.
2. **Data Processing:** Pre-flight validation and conversion of COCO dataset to YOLO format and EDA.
3. **Model Training:** (In progress)
4. **Deployment:** (In progress)

## Pipeline Design
Each stage of the pipeline is defined by an abstract base class. Supporting a new source or target format means implementing a new interface:

| Contract | Responsibility | Current implementation |
| --- | --- | --- |
| `BaseFormatHandler` | Discovers annotation files per split and loads them | `CocoFormatHandler` |
| `BaseAnnotationConverter` | Converts one annotation file into image path and label pairs | `CocoToYoloDetectionConverter` |
| `BaseDatasetWriter` | Writes images, labels, and format metadata to disk | `YoloWriter` |
| `BaseDatasetValidator` | Runs the pre-flight checks for a source format | `CocoDatasetValidator` |

### Pre-flight Validation
The dataset is validated before any file is written to disk. The validator collects all check failures instead of immediately aborting, writes them to `logs/validation.log`, and stops the pipeline if any are found:

1. **Annotation structure:** `images` and `categories` present and non-empty; non-test splits declare annotations
2. **Class consistency:** every annotation file defines identical class id/name pairs
3. **Class mapping coverage:** the configured class mapping and the annotation categories match exactly
4. **Annotation completeness:** unique image and annotation ids, unique image paths, every image annotated, no orphaned annotation references
5. **Image dimensions:** every image declares a positive integer width and height
6. **Filename uniqueness:** no two annotation files within a split reference the same output filename (LOCO includes multiple annotation files per split)
7. **Missing images:** every referenced image path exists on disk
8. **Bounding box validity:** four numeric coordinates with positive width and height
9. **Data leakage:** no image appears in more than one split

**Note:** This validation suite describes the current COCO implementation.

## Repository Structure

```text
├── .github/
│   └── workflows/
│       └── gates.yml        # CI gates: ruff, mypy, and the unit suite
├── configs/                 # Dataset ingestion and processing settings
├── data/
│   ├── external/            # Data from third party sources (Git-ignored)
│   ├── processed/           # Normalised YOLO format data (Git-ignored)
│   └── raw/                 # Original COCO datasets (Git-ignored)
├── dvc.lock                 # Resolved stage dependency and output hashes
├── dvc.yaml                 # DVC pipeline stage definitions
├── init_mlops.sh            # Environment and tracking initialisation script
├── LICENSE
├── models/                  # Trained models, ONNX exports, and model summaries
├── notebooks/               # Sandboxed Exploratory Data Analysis (EDA)
├── pyproject.toml           # Project metadata, ruff and mypy configuration
├── README.md                # Project documentation
├── requirements-base-dev.txt# Testing, typing, and EDA dependencies
├── requirements-base.txt    # Core dependencies for data engineering pipeline
├── src/                     # Production-ready pipeline source code
│   ├── api/                 # Microservice deployment code (FastAPI)
│   ├── converters/          # Annotation conversion logic (e.g., COCO to YOLO)
│   ├── data/                # Data ingestion and processing orchestrators
│   │   ├── ingest_data.py
│   │   └── process_dataset.py
│   ├── datatypes.py         # Shared type aliases and pydantic contracts
│   ├── handlers/            # Source format file discovery and loading
│   ├── models/              # Model training and evaluation scripts
│   ├── paths.py             # Global path definitions
│   ├── utils/               # Shared helper functions
│   │   ├── coco_ops.py
│   │   └── file_ops.py
│   ├── validators/          # Pre-flight dataset validation
│   └── writers/             # Target format dataset writers
└── tests/                   # Pytest suite
    ├── integration/         # Live DagsHub / MLflow integration tests
    └── unit/                # Unit tests, mirroring the src/ package layout
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
The raw data is version-controlled as an immutable artifact and is retrieved from the project's DVC remote. `init_mlops.sh` configures that remote from the values in `.env`.

```bash
dvc pull
```

### Manual Data Ingestion (Rebuild from Source)
*The raw ingestion script rebuilds the dataset from scratch, downloading LOCO directly from the TUM servers. This is the route taken when updating the dataset or rebuilding the DVC pipeline.*

*Credits to the TUM team for creating and supplying the dataset in their repository:* https://github.com/tum-fml/loco

>LOCO: Logistics Objects in Context    
>Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J.     
>IEEE International Conference on Machine Learning and Applications (ICMLA) 2020

```bash
python -m src.data.ingest_data --dataset loco --config configs/data_config.yaml
```

### Data Processing
The COCO to YOLO conversion runs a pre-flight validation pass over the raw dataset, and an inconsistent dataset fails before any output is written. The conversion is declared as a DVC stage (`process_loco`) in `dvc.yaml`.

Reproduce the stage from its tracked dependencies:

```bash
dvc repro
```

Or invoke the converter directly, bypassing DVC:

```bash
python -m src.data.process_dataset --dataset loco --config configs/data_config.yaml
```

The LOCO run produces 2820 training and 2277 validation image-label pairs across 5 classes, along with a root configuration file used by YOLO (`dataset.yaml`).

Validation errors are written to `logs/validation.log`. Progress and warnings go to the terminal.

### Exploratory Data Analysis (EDA)
All exploratory notebooks are isolated in the `notebooks/` directory to prevent environment pollution. To run the EDA notebooks, ensure you have installed the development dependencies using `requirements-base-dev.txt`.

From within the conda environment launch `marimo`:

```bash
marimo edit notebooks/01_EDA_raw_loco.py
```

## Testing
Unit and pipeline integration tests have been built to verify the correct execution of the utility functions, experiment tracking, and processing logic.

Additionally, `mypy` is used for ensuring typing adherence and `ruff` for linting and import sorting. Both tools are configured in `pyproject.toml`. All three quality gates are integrated into a CI pipeline and run on every push and pull request via GitHub Actions.

```bash
# Lint and verify import sorting
ruff check src/ tests/

# Apply safe automatic fixes (import sorting, etc.)
ruff check --fix src/ tests/

# Verify strict typing
mypy src/ tests/

# Run the unit test suite
pytest tests/unit
```

Every project data directory and the validation log path are redirected into a temporary directory for the duration of each unit test, so the suite never reads or writes to the real `data/` or `logs/` directories. All outbound requests are mocked, so no test touches the network.

The integration suite connects to the live DagsHub MLflow server and is only run manually:

```bash
pytest tests/integration
```

Unit test branch coverage is measured with `pytest-cov` and currently sits at **92%** across `src/`:

```bash
pytest tests/unit --cov=src --cov-branch --cov-report=term-missing
```