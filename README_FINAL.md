# Flavor Scout

AI-powered consumer trend analytics dashboard for discovering data-backed flavor opportunities across HealthKart brands (MuscleBlaze, HK Vitals, TrueBasics).

The tool ingests social-style consumer comments, extracts trending flavors, analyzes sentiment, and generates prioritized recommendations with clear business rationale.

## Project Overview

Flavor Scout is a Streamlit web application that processes a curated dataset of 119 consumer comments (Reddit-style reviews focused on taste and flavor preferences). It uses an LLM (Groq) to:

- Extract relevant flavor mentions
- Compute mention frequency and sentiment
- Rank flavors by consumer demand and positivity
- Generate brand-aligned recommendations
- Highlight a single top recommendation with a concise business justification

The application operates entirely in demo mode using mock data — no live API credentials are required for social platform ingestion.

## Architecture

The project follows a modular, maintainable structure:

- **app.py**  
  Main Streamlit application. Handles UI layout, user inputs (query, brand selection, data volume), caching, and orchestration of the analysis pipeline.

- **ai_analysis.py**  
  Core reasoning engine. Contains LLM prompts and logic for flavor extraction, sentiment analysis, ranking, and generation of recommendation explanations. No core prompts or ranking rules were modified during development.

- **data_ingestion.py**  
  Responsible for loading the mock dataset (mock_data.json). Provides a consistent interface for retrieving the full set of 119 comments.

- **mock_data.json**  
  Curated dataset of 119 authentic consumer comments focused on flavor and taste experiences.

- **requirements.txt**  
  Complete list of Python dependencies required to run the application.

- **tests/** (various test scripts)  
  Includes unit and integration tests for data loading, trend extraction, brand alignment, and pipeline verification.

## How It Works

1. User configures analysis via sidebar:
   - Optional search query (influences LLM attention, not data filtering)
   - Target brand (MuscleBlaze, HK Vitals, TrueBasics, or All)
   - Data volume slider (controls sample size for analysis)

2. On "Analyze Trends":
   - Full dataset is loaded
   - LLM extracts flavor mentions and assesses sentiment
   - Frequencies and sentiment scores are calculated
   - Flavors are ranked by a combination of mention volume and positive sentiment

3. Results displayed:
   - Consumer Signal Overview (key metrics)
   - Flavor Opportunity Map (bar chart of top mentions)
   - Recommendation Engine (strong candidates vs lower-priority flavors)
   - Top Recommendation (single highest-ranked flavor with business rationale)
   - Raw data samples (top relevant comments)

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
