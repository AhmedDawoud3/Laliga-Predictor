# Football Match Predictor

This project includes components for scraping football match data, cleaning and processing the data, building a machine learning model to predict match outcomes, and a Dash web application for interaction with the model.

![image](https://github.com/AhmedDawoud3/Laliga-Predictor/assets/68483546/c2e86184-e294-4155-9559-a8b7612f043e)

## Components

### 1. Scraper

The script (`scraper.py`) responsible for fetching football match data from [soccerstats](soccerstats.com/latest.asp?league=spain) for the last 7 seasons.

### 2. Cleaner

The script (`cleaner.py`) which cleans and preprocesses the scraped data. It creates structured tables and adds features like team form to prepare the data for modeling.

### 3. Model

The model script (`model.py`) that builds a RandomForestClassifier pipeline. It trains the model using the cleaned data and saves the trained model (`model.pkl`) for later use.

### 4. Dash Web Application

The `app.py` file is a Dash web application that provides an interface for users to interact with the trained model. Users can select home and away teams, adjust their form, and make predictions on match outcomes through a user-friendly web interface.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedDawoud3/Laliga-Predictor
   cd Laliga-Predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Scraper
Run the scraper script to fetch football match data:

```bash
python .\scraper.py <output_csv>
```
* This saves a `csv` with all the matches for the past 7 seasons

2. Cleaner
Clean, preprocess, and creates the tables for the 7 season for the scraped data using the cleaner script:
```bash
python .\cleaner.py <csv_file> <output_folder>
```
* This saves the cleaned data to a file with 'cleaned' suffix
* `saved_seasons.csv` -> `saved_seasons_cleaned.csv`
3. Model
Build the machine learning model using the model script:
```bash
python .\model.py <cleaned_csv> <model_output>
```
* This saves the model to `model_output.pkl` so it can be used in the web app
4. Dash Web Application
Launch the Dash web application to interact with the trained model:
```bash
python .\app.py <cleaned_csv> <saved_model_pkl>
```
* Open your web browser and navigate to [http://127.0.0.1:8050/](http://127.0.0.1:8050/) to access the application.


  
# Contributing
Contributions are welcome! Please feel free to fork the repository, make pull requests, or open issues for any bugs or feature requests.
