import logging
import os
import sys

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def wrangle(filename: str) -> pd.DataFrame:
    """
    Processes the input CSV file and prepares the data for analysis.

    Args:
        filename (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = pd.read_csv(filename)

    # Extract home and away scores
    df["home_score"] = df["score"].str.split(" - ").str[0].astype(int)
    df["away_score"] = df["score"].str.split(" - ").str[1].astype(int)

    # Extract half-time scores
    df["HT_home_score"] = df["HT"].str.split("-").str[0].str[1:].astype(int)
    df["HT_away_score"] = df["HT"].str.split("-").str[1].str[:-1].astype(int)

    df.drop(columns=["score", "HT"], inplace=True)

    # Determine match winner
    df["winner"] = df.apply(
        lambda x: (
            "home"
            if x["home_score"] > x["away_score"]
            else "away" if x["away_score"] > x["home_score"] else "draw"
        ),
        axis=1,
    )

    return df


def calculate_season_table(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Calculates the league table for a given season.

    Args:
        df (pd.DataFrame): The DataFrame containing match data.
        season (str): The season to calculate the table for.

    Returns:
        pd.DataFrame: The league table for the season.
    """
    logging.info("Calculating table for season: %s", season)
    season_df = df[df["Season"] == season].reset_index(drop=True)

    teams = pd.concat([season_df["home"], season_df["away"]]).unique()
    table = pd.DataFrame(teams, columns=["Team"])

    # Initialize table columns
    table_stats = ["MP", "W", "D", "L", "GF", "GA", "GD", "PTS"]
    for stat in table_stats:
        table[stat] = 0

    for _, match in season_df.iterrows():
        home_team = match["home"]
        away_team = match["away"]
        home_score = match["home_score"]
        away_score = match["away_score"]

        # Update matches played
        table.loc[table["Team"] == home_team, "MP"] += 1  # type: ignore
        table.loc[table["Team"] == away_team, "MP"] += 1  # type: ignore

        # Update win, draw, loss, points
        if home_score > away_score:
            table.loc[table["Team"] == home_team, "W"] += 1  # type: ignore
            table.loc[table["Team"] == away_team, "L"] += 1  # type: ignore
            table.loc[table["Team"] == home_team, "PTS"] += 3  # type: ignore
        elif home_score < away_score:
            table.loc[table["Team"] == away_team, "W"] += 1  # type: ignore
            table.loc[table["Team"] == home_team, "L"] += 1  # type: ignore
            table.loc[table["Team"] == away_team, "PTS"] += 3  # type: ignore
        else:
            table.loc[table["Team"] == home_team, "D"] += 1  # type: ignore
            table.loc[table["Team"] == away_team, "D"] += 1  # type: ignore
            table.loc[table["Team"] == home_team, "PTS"] += 1  # type: ignore
            table.loc[table["Team"] == away_team, "PTS"] += 1  # type: ignore

        # Update goals for, goals against, goal difference
        table.loc[table["Team"] == home_team, "GF"] += home_score
        table.loc[table["Team"] == away_team, "GF"] += away_score
        table.loc[table["Team"] == home_team, "GA"] += away_score
        table.loc[table["Team"] == away_team, "GA"] += home_score
        table.loc[table["Team"] == home_team, "GD"] += home_score - away_score
        table.loc[table["Team"] == away_team, "GD"] += away_score - home_score

    table.sort_values(by=["PTS", "GD", "GF"], ascending=False, inplace=True)
    table.reset_index(drop=True, inplace=True)
    table["position"] = table.index + 1

    # Reorder columns to make position the first column
    table = table[["position"] + [col for col in table.columns if col != "position"]]

    return table


def process_teams(df: pd.DataFrame) -> dict:
    """
    Processes each team's home and away results and adds the result of each match.

    Args:
        df (pd.DataFrame): The DataFrame containing match data.

    Returns:
        dict: A dictionary containing processed DataFrames for each team's home and away matches.
    """
    teams = {}
    for team in df["home"].unique():
        teams[f"{team}_home"] = df[df["home"] == team].copy()
        teams[f"{team}_home"]["result"] = np.where(
            teams[f"{team}_home"]["winner"] == "home",
            "won",
            np.where(teams[f"{team}_home"]["winner"] == "away", "lost", "draw"),
        )
        teams[f"{team}_away"] = df[df["away"] == team].copy()
        teams[f"{team}_away"]["result"] = np.where(
            teams[f"{team}_away"]["winner"] == "away",
            "won",
            np.where(teams[f"{team}_away"]["winner"] == "home", "lost", "draw"),
        )

    return teams


def add_team_form(df: pd.DataFrame, teams: dict) -> None:
    """
    Adds home and away form (previous 5 matches) for each team to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing match data.
        teams (dict): The dictionary containing processed DataFrames for each team's home and away matches.
    """
    df["home_form"] = 0
    df["away_form"] = 0
    for team, team_df in teams.items():
        if "home" in team:
            df.loc[team_df.index, "home_form"] = (
                team_df["result"]
                .apply(lambda x: 1 if x == "won" else 0)
                .rolling(5)
                .sum()
            )
        elif "away" in team:
            df.loc[team_df.index, "away_form"] = (
                team_df["result"]
                .apply(lambda x: 1 if x == "won" else 0)
                .rolling(5)
                .sum()
            )


def main(input_csv: str, output_folder_path: str):
    """
    Main function to process the input file and generate league tables for each season.

    Args:
        input_csv (str): The path to the input CSV file.
        output_folder_path (str): The path to the output folder.
    """
    logging.info("Processing file: %s", input_csv)
    df = wrangle(input_csv)
    teams = process_teams(df)

    # Add home and away form to the DataFrame
    add_team_form(df, teams)

    cleaned_file = f"{os.path.splitext(input_csv)[0]}_cleaned.csv"
    logging.info("Saving cleaned data to %s", f"{input_csv[:-4]}_cleaned.csv")
    df.to_csv(cleaned_file, index=False)

    seasons = df["Season"].unique()
    os.makedirs(output_folder_path, exist_ok=True)

    for season in seasons:
        season_table = calculate_season_table(df, season)
        output_file = os.path.join(
            output_folder_path, f"{season.replace('/', '_')}.csv"
        )
        season_table.to_csv(output_file, index=False)
        logging.info("Saved table for season %s to %s", season, output_file)

    logging.info("Processing complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python script.py <input_csv> <output_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    main(input_file, output_folder)
