# FIFA World Cup Winner Prediction

> **6th Semester University Lab Project**

A machine learning project that estimates football match outcomes and simulates a possible FIFA World Cup tournament using historical international match data.

## Project Overview

This project combines historical results, team performance statistics, and Elo ratings to estimate the probability of:

- Team 1 winning
- A draw
- Team 2 winning

It then uses those probabilities to simulate group-stage matches and knockout rounds for a 48-team 2026 World Cup format.

## Machine Learning Approach

The prediction pipeline includes:

- Historical international match results from `results.csv`
- Competitive-match and recent-form statistics from 2015 onward
- Chronological Elo rating updates
- Gradient Boosting classification
- Feature scaling with `StandardScaler`
- Train/test evaluation using accuracy, precision, recall, and F1 score
- A draw-probability adjustment based on Elo difference

## Tournament Simulation

The script automatically:

1. Ranks the selected 48 teams using Elo and recent form.
2. Creates 12 groups with 4 teams each.
3. Simulates all group-stage fixtures.
4. Selects group winners, runners-up, and the best third-place teams.
5. Simulates the Round of 32, Round of 16, quarter-finals, semi-finals, and final.
6. Provides a predicted tournament winner.
7. Runs a custom match prediction for two teams entered by the user.

## Dataset

`results.csv` contains historical international football matches with fields including:

- Date
- Home team and away team
- Home score and away score
- Tournament
- Match location
- Neutral-ground indicator

## Requirements

- Python 3.10+
- pandas
- NumPy
- scikit-learn

Install the dependencies with:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

From the repository root:

```bash
python "FIFA World Cup Winner Prediction/FIFA.py"
```

The program trains the model, prints evaluation metrics, simulates the tournament, and asks for two teams for a custom prediction.

Example input:

```text
Brazil
France
```

## Example Output

The output includes:

- Number of matches loaded
- Number of teams with calculated statistics
- Model evaluation metrics
- Strongest-team ranking
- Simulated groups and standings
- Knockout-round winners
- Predicted World Cup winner
- Custom match probabilities

## Academic Note

This project was developed as part of the **6th Semester University Laboratory Projects**. The prediction is educational and statistical; it is not an official FIFA forecast or a guarantee of real-world match results.
