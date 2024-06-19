import os
import pickle
import sys

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output


# Function to load the model from a pickle file
def load_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError(
            "The loaded object is not a trained model or does not support probability predictions."
        )

    return model


# Function to predict the outcome of a match
def predict_outcome(home_team, away_team, home_form, away_form, model):
    # Prepare the input data for prediction
    input_data = pd.DataFrame(
        {
            "home": [home_team],
            "away": [away_team],
            "home_form": [home_form if home_form is not None else 0],
            "away_form": [away_form if away_form is not None else 0],
        }
    )

    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        home_win_prob = probabilities[2] * 100
        draw_prob = probabilities[1] * 100
        away_win_prob = probabilities[0] * 100

        outcome = (
            f"The predicted outcome is: {home_team} wins"
            if prediction == 1
            else (
                f"The predicted outcome is: {away_team} wins"
                if prediction == -1
                else "The predicted outcome is: Draw"
            )
        )

        details = html.Div(
            [
                html.H2(outcome, style={"fontSize": "24px", "color": "#3498db"}),
                html.P(f"{home_team} wins by {home_win_prob:.2f}% chance."),
                html.P(f"The teams draw by {draw_prob:.2f}% chance."),
                html.P(f"{away_team} wins by {away_win_prob:.2f}% chance."),
            ]
        )
        return details
    except Exception as e:
        return f"Error during prediction: {str(e)}"


# Function to initialize the Dash app
def initialize_app():
    app = dash.Dash(__name__)
    app.title = "Football Match Predictor"
    return app


# Function to define the layout of the app
def setup_layout(app):
    layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "maxWidth": "800px",
            "margin": "auto",
        },
        children=[
            html.Div(
                style={"textAlign": "center", "marginBottom": "20px"},
                children=[
                    html.Img(
                        src=app.get_asset_url("LL_RGB_h_color.png"),
                        style={"width": "200px", "height": "auto"},
                    )
                ],
            ),
            html.H1(
                "Football Match Outcome Predictor",
                style={
                    "textAlign": "center",
                    "color": "#2c3e50",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                style={"marginBottom": "20px"},
                children=[
                    html.Label(
                        "Home Team", style={"fontWeight": "bold", "color": "#2c3e50"}
                    ),
                    dcc.Dropdown(
                        id="home-team",
                        options=[],
                        placeholder="Select Home Team",
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
            html.Div(
                style={"marginBottom": "20px"},
                children=[
                    html.Label(
                        "Away Team", style={"fontWeight": "bold", "color": "#2c3e50"}
                    ),
                    dcc.Dropdown(
                        id="away-team",
                        options=[],
                        placeholder="Select Away Team",
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
            html.Div(
                style={
                    "marginBottom": "20px",
                    "display": "flex",
                    "justifyContent": "space-between",
                },
                children=[
                    html.Div(
                        style={"flex": "1", "marginRight": "10px"},
                        children=[
                            html.Label(
                                "Home Form (Number of matches won in the last 5)",
                                style={"fontWeight": "bold", "color": "#2c3e50"},
                            ),
                            dcc.Slider(
                                id="home-form",
                                min=0,
                                max=5,
                                step=1,
                                marks={i: str(i) for i in range(6)},
                                value=0,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1", "marginLeft": "10px"},
                        children=[
                            html.Label(
                                "Away Form (Number of matches won in the last 5)",
                                style={"fontWeight": "bold", "color": "#2c3e50"},
                            ),
                            dcc.Slider(
                                id="away-form",
                                min=0,
                                max=5,
                                step=1,
                                marks={i: str(i) for i in range(6)},
                                value=0,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={"textAlign": "center", "marginBottom": "20px"},
                children=[
                    html.Button(
                        "Predict Outcome",
                        id="predict-button",
                        style={
                            "backgroundColor": "#3498db",
                            "color": "white",
                            "border": "none",
                            "padding": "10px 20px",
                            "cursor": "pointer",
                            "fontSize": "16px",
                        },
                    ),
                ],
            ),
            html.Div(
                id="prediction-result",
                style={
                    "textAlign": "center",
                    "color": "#e74c3c",
                    "fontSize": "20px",
                    "fontWeight": "bold",
                },
            ),
        ],
    )
    return layout


# Function to set team options in the dropdowns
def set_team_options(df):
    teams = sorted(df["home"].unique())
    return [{"label": team, "value": team} for team in teams]


# Function to define the callbacks
def setup_callbacks(app, df, model):
    @app.callback(
        [Output("home-team", "options"), Output("away-team", "options")],
        [Input("home-team", "options"), Input("away-team", "options")],
    )
    def set_team_options_callback(home_options, away_options):
        return set_team_options(df), set_team_options(df)

    @app.callback(
        Output("prediction-result", "children"),
        Input("predict-button", "n_clicks"),
        [
            Input("home-team", "value"),
            Input("away-team", "value"),
            Input("home-form", "value"),
            Input("away-form", "value"),
        ],
    )
    def update_prediction_result(n_clicks, home_team, away_team, home_form, away_form):
        if n_clicks is None:
            return ""

        if not home_team or not away_team:
            return "Please select both home and away teams."

        prediction = predict_outcome(home_team, away_team, home_form, away_form, model)

        return html.Div(
            [
                html.H2(
                    "Prediction Result", style={"fontSize": "24px", "color": "#3498db"}
                ),
                html.P(prediction),
            ]
        )


# Function to run the Dash app
def run_app(app):
    app.run_server(debug=True)


# Main function to orchestrate the entire application
def main(csv_file, model_filepath):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Load the model
    model = load_model(model_filepath)

    # Initialize the Dash app
    app = initialize_app()

    # Setup the layout
    layout = setup_layout(app)
    app.layout = layout

    # Setup callbacks
    setup_callbacks(app, df, model)

    # Run the app
    run_app(app)


# Entry point of the application
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file> <model_filepath>")
        sys.exit(1)

    csv_file = sys.argv[1]
    model_filepath = sys.argv[2]

    # Execute the main function
    main(csv_file, model_filepath)
