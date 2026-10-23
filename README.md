# BTC Sentiment Price Prediction Agent

A Python machine-learning project that explores the relationship between Bitcoin (BTC) market prices and social-media sentiment.

## Features
- Tweet sentiment preparation and daily aggregation
- Historical Bitcoin price data processing
- Feature and target alignment checks
- Random Forest price prediction
- Walk-forward validation against a naive baseline
- Sentiment-versus-price analysis and strategy evaluation
- Evaluation report in PDF and Markdown formats

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Project Structure
- `src/` — data processing, model training, validation, visualization, and strategy evaluation scripts
- `data/` — datasets, predictions, and analysis outputs
- `evaluation_report.pdf` — model evaluation report
- `evaluation_report.md` — Markdown version of the report

## How to Run
Activate your virtual environment, install the required libraries, and run the scripts from the project root. The scripts use files in the `data/` directory, so run them in the documented pipeline order.

## Evaluation
Walk-forward validation showed that the Random Forest model did not outperform the naive next-day closing-price baseline on MAE or RMSE. Results are experimental and do not establish profitable trading performance.

## Limitations
The sentiment dataset used for the historical analysis covers 2022–2023. Current social-media/news integration and broader real-time validation remain future improvements.

## Author
Aman Kumar
