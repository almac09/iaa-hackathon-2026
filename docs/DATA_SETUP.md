# 📦 Data Setup Guide — IAA Hackathon 2026

This guide walks you through downloading the competition data from Kaggle and loading it into Databricks.

## Competition Link

🔗 **https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1**

Make sure you've joined the competition on Kaggle before downloading data.

---

## Step 1: Download Data from Kaggle

### Option A: Manual Download (easiest)

1. Go to the [competition Data tab](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/data)
2. Click **Download All** to get a zip file
3. Extract the zip — you should have CSV files (typically `train.csv`, `test.csv`, and `sample_submission.csv`)

### Option B: Kaggle CLI (if you have it installed)

```bash
# Install Kaggle CLI (one-time)
pip install kaggle

# Set up your API token:
# Go to kaggle.com → Your Profile → Account → Create New API Token
# Save kaggle.json to ~/.kaggle/kaggle.json

# Download competition data
kaggle competitions download -c iaa-ai-assisted-hackathon-2026-part-1
unzip iaa-ai-assisted-hackathon-2026-part-1.zip
```

### Option C: Kaggle CLI from a Databricks Notebook

You can run the Kaggle CLI directly in a Databricks notebook:

```python
# Install kaggle CLI
%pip install kaggle

# Set your Kaggle credentials (get these from kaggle.com → Account → API)
import os
os.environ['KAGGLE_USERNAME'] = '<your_kaggle_username>'
os.environ['KAGGLE_KEY'] = '<your_kaggle_api_key>'

# Download to a temporary location
!kaggle competitions download -c iaa-ai-assisted-hackathon-2026-part-1 -p /tmp/kaggle_data
!unzip -o /tmp/kaggle_data/iaa-ai-assisted-hackathon-2026-part-1.zip -d /tmp/kaggle_data/

# Copy to your Unity Catalog Volume
import shutil
volume_path = "/Volumes/<your_catalog>/<your_schema>/<your_volume>/hackathon"
dbutils.fs.mkdirs(volume_path)

# List extracted files
import os
for f in os.listdir('/tmp/kaggle_data/'):
    print(f)
    if f.endswith('.csv'):
        shutil.copy(f'/tmp/kaggle_data/{f}', volume_path.replace('/Volumes', '/Volumes') + f'/{f}')
```

> ⚠️ **Security note**: Never commit your Kaggle API key to Git. Use Databricks Secrets or environment variables.

---

## Step 2: Upload to Databricks

### Option A: Drag & Drop Upload (easiest)

1. In Databricks, click **New → Add or upload data**
2. Click **Upload files to a volume**
3. Select your catalog, schema, and volume (or create a new volume)
4. Drag and drop your CSV files
5. Note the volume path — you'll need it for the notebooks

### Option B: Create a Table Directly

1. In Databricks, click **New → Add or upload data**
2. Click **Create or modify a table using file upload**
3. Upload your `train.csv`
4. Databricks will auto-detect the schema
5. Choose a catalog and schema, name the table (e.g., `hackathon_train`)
6. Click **Create table**
7. Repeat for `test.csv`

---

## Step 3: Update the Notebooks

Once your data is uploaded, update the paths in the notebooks:

### If you uploaded to a Volume:

Open `notebooks/01_data_loading.py` and update:

```python
VOLUME_PATH = "/Volumes/<your_catalog>/<your_schema>/<your_volume>/hackathon"
```

### If you created tables directly:

You can skip notebook 01 and load directly in notebook 02:

```python
train_df = spark.read.table("<your_catalog>.<your_schema>.hackathon_train")
test_df = spark.read.table("<your_catalog>.<your_schema>.hackathon_test")
```

---

## Step 4: Verify the Data

Run these checks in a notebook to make sure everything loaded correctly:

```python
# Check row counts
print(f"Train rows: {train_df.count()}")
print(f"Test rows: {test_df.count()}")

# Check schema
train_df.printSchema()

# Preview data
display(train_df.limit(5))
```

---

## Folder Structure After Setup

Your Unity Catalog Volume should look like:

```
/Volumes/<catalog>/<schema>/<volume>/hackathon/
├── train.csv
├── test.csv
└── sample_submission.csv
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Access denied" on Volume | Check you have WRITE VOLUME permission — ask your workspace admin |
| CSV won't upload | Make sure file is under 5GB; for larger files use Kaggle CLI method |
| Schema looks wrong | Try `inferSchema=True` or manually specify the schema |
| Can't find the competition | Make sure you've joined it at the Kaggle link above |

---

## Next Steps

Once data is loaded and verified:
1. ✅ Close Issue #1 (Data Loading) on the project board
2. Move to `notebooks/02_eda.py` for Exploratory Data Analysis
3. Log your data loading decisions in `notebooks/00_decision_log.py`
