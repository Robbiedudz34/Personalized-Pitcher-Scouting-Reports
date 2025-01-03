# MLB Batter-Pitcher Scouting Report Generator

This Python script generates a detailed scouting report for a specific batter-pitcher matchup based on 2024 season pitch-level data. The report includes summaries, heatmaps, and key insights to assist in analyzing player performance. Download the zip file to access the data that I used in this program, and save to the same location as your script.

## Features
- Analyze specific batter-pitcher combinations.
- Generate detailed PDF reports with key metrics, heatmaps, and summaries.
- Dynamic database integration to map player IDs to names.

## Requirements

Ensure the following Python packages are installed:
- pandas
- numpy
- matplotlib
- seaborn
- reportlab
- PyPDF2
- mysql-connector-python
- python-dotenv (optional, for handling credentials securely)

You can install all dependencies with:
`bash
pip install -r requirements.txt

You will need to create a database with MySQL that connects all player ID numbers to their names. This database is essential for the script to function correctly. Once the database is set up, you can initialize the script with your MySQL credentials (user, password, host, and database).

#### Step 1: Install MySQL
Ensure you have MySQL installed on your machine. You can download it from [MySQL Community Downloads](https://dev.mysql.com/downloads/).

#### Step 2: Create the Database
Use the following script to set up the database and the players table. This script creates a new database, a players table, and inserts two players.


sql
CREATE DATABASE mlbdata;

USE mlbdata;

CREATE TABLE players (
    player_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

INSERT INTO players (player_id, name) VALUES
(467793, 'Carlos Santana'),
(666142, 'Cole Ragans');


Note - the default players in the script are from these numbers. I'll leave it to you to create the entire list of players with their correct combinations.

Running in the terminal will prompt for a specific batter or pitcher name. If no characters are entered, a default matchup has been placed.
