import logging
import re
import sys
import warnings
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# URLS
LATEST_URL = "https://www.soccerstats.com/latest.asp?league=spain"
BASE_URL = "https://www.soccerstats.com/"
RESULT_URL = "https://www.soccerstats.com/results.asp?league=spain&pmtype=bydate"

# logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_page(
    url: str, retries: int = 3, timeout: int = 5
) -> Optional[requests.Response]:
    """
    Fetches the content of a web page.

    Args:
        url (str): The URL of the web page to fetch.
        retries (int, optional): The number of retries in case of request failure. Defaults to 3.
        timeout (int, optional): The timeout for the request in seconds. Defaults to 5.

    Returns:
        Optional[requests.Response]: The response object if the request is successful, otherwise None.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.error("Error fetching %s: %s", url, e)
            if attempt < retries - 1:
                sleep(2**attempt)
            else:
                logging.error("Failed to fetch %s after %d attempts", url, retries)
                return None


def parse_table(soup: BeautifulSoup, table_id: str, table_text: str) -> pd.DataFrame:
    """
    Parses a table from a BeautifulSoup object based on the given table ID and table text.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.
        table_id (str): The ID of the table to be parsed.
        table_text (str): The text that should be present in the table.

    Returns:
        pandas.DataFrame: The parsed table as a DataFrame.

    Raises:
        ValueError: If the table with the specified text is not found.
    """
    table = [i for i in soup.select(table_id) if table_text in str(i)]
    if table:
        return pd.read_html(str(table[0]))[0]
    else:
        raise ValueError(f"Table with text '{table_text}' not found.")


def scrape_season(url: str, season_name: str) -> pd.DataFrame:
    """
    Scrapes the data for a given season from a specified URL.

    Args:
        url (str): The URL of the webpage containing the data.
        season_name (str): The name of the season.

    Returns:
        pd.DataFrame: A DataFrame containing the scraped data.
    """
    logging.info("Scraping %s, url = %s", season_name, url)
    response = fetch_page(url)
    if response is None:
        raise ValueError(f"Failed to fetch data for season: {season_name}")

    soup = BeautifulSoup(response.text, "html.parser")
    table = parse_table(soup, "#btable", "Half-time score")
    df = table.iloc[1:, :-4]
    df.columns = ["date", "home", "score", "away", "DEL", "HT"]
    df.dropna(subset=["date"], inplace=True)
    df.drop(columns=["DEL"], inplace=True)
    df["Season"] = season_name
    df.reset_index(drop=True, inplace=True)
    logging.info("Scraped %s", season_name)
    return df


def get_season_urls(matches_soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """
    Extracts season URLs and names from the matches page soup.

    Args:
        matches_soup (BeautifulSoup): The BeautifulSoup object containing the matches page HTML.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the season URLs and their names.
    """
    seasons = []
    dropdowns = matches_soup.find_all(class_="dropdown")
    for dropdown in dropdowns:
        if "latest.asp?league=spain" in str(dropdown):
            links = dropdown.find_all("a")
            for link in links:
                href = link.get("href")
                text = link.get_text().strip()
                if "league=spain" in href and not re.search(r"[A-Za-z]", text):
                    season_url = (
                        BASE_URL + href.replace("latest", "results") + "&pmtype=bydate"
                    )
                    seasons.append((season_url, text))
    return seasons


def main(filename: str = "all_seasons.csv"):
    logging.info("Scraping soccerstats.com")

    logging.info("Fetching matches data")
    # Fetch the latest matches data

    matches_data = fetch_page(RESULT_URL)
    if matches_data is None:
        logging.error("Failed to fetch matches data")
        return
    matches_soup = BeautifulSoup(matches_data.text, "html.parser")

    # Get all seasons
    seasons = get_season_urls(matches_soup)

    # Scrape each season to a DataFrame
    seasons_dfs = []
    for season_url, season_name in seasons:
        try:
            season_df = scrape_season(season_url, season_name)
            seasons_dfs.append(season_df)
            sleep(1)  # Rate limiting
        except ValueError as e:
            logging.error("Failed to scrape %s: %s", season_name, e)

    # Concatenate all season DataFrames
    if seasons_dfs:
        all_seasons_df = pd.concat(seasons_dfs)
        all_seasons_df.reset_index(drop=True, inplace=True)

        # Save the DataFrame to CSV
        output_file = f"{filename}.csv"
        all_seasons_df.to_csv(output_file, index=False)
        logging.info("Data saved to %s", output_file)
    else:
        logging.warning("No data to save")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    main(sys.argv[1])
