# Predictor of nearest apartments available for renting on QuintoAndar website based on user-provided information

This repository contains a Python pipeline to find houses similar to a user-provided input using K-Means clustering and K-Nearest Neighbors (KNN) algorithm. It includes data wrangling, scaling, dimensionality reduction (PCA), clustering, visualization, and similarity search.

---

## Project Structure

```
├── config
    ├── user_input.json       # user-provided features of "ideal" house
├── database                  # data base where scrapped data and urls are saved
    ├── data                  # data from apartments for renting scraped from the QuintoAndar website
    ├── urls                  # urls folder
├── modules                   # contains essencial functions for clustering, KNN, and plotting
├── tables                    # contains output table from KNN with the 10 nearest apartments based on the user-provided data
├── main.py                   # main application entry point
├── requirements.txt          # python dependencies
└── README.md                 # this file
```

## Overview

This application is intended to predict the nearest houses available for renting in five neighboods of Sao Paulo:
- **1**: Pinheiros
- **2**: Alto de Pinheiros
- **3**: Jaguare
- **4**: Butanta
- **5**: Vila Madalena

Those datasets were obtained from the QuintoAndar website (https://www.quintoandar.com.br/), using the scraper developed by Dr. Jose Devienne: https://github.com/devienne/Tech-Challenge-QuintoAndar

## Features

- **Data Wrangling**: Clean and preprocess real estate CSV datasets.
- **Scaling & Transformation**: Apply `ColumnTransformer` for numeric and binary features.
- **K-Means Clustering**: Cluster properties into groups for comparison.
- **KNN Similarity Search**: Find the most similar houses within the same cluster.
- **Visualization**:
  - PCA 2D plots for clusters with decision regions and centroids.
  - Side-by-side visualization of user input and nearest neighbors.
- **Configurable User Input**: Load user preferences from a JSON configuration file.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ReginaldoAllves/Finding_Similar_Houses.git
cd Finding_Similar_Houses
```

2. Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data: Place CSV files in ./database/data/. Ensure consistent column naming.
2. Configure user input:
   Run the streamlit app
   ```bash
    streamlit run app.py
    ```
   Open the browser and go to:
   ```bash
    http://localhost:8501
    ```

## Run the pipeline

After filling in the information that you want at the input area, press the "Enviar dados" button.

## Retrieve nearest apartments in relation to user-provided information

To check on the features (url, rent, etc ...) of the 25 nearest apartments for renting, based on the user-provided information, open a terminal and run:
   ```bash
    python -m http.server 8000
   ```
   Open the browser and go to:
   ```bash
    http://0.0.0.0:8000
   ```

## Disclaimer

This application is intended to be used in academic and research scenarios only.

## Contact

Dr. Reginaldo C. A. Rosa
- **Email**: reginaldocarosa@gmail.com
