import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.platypus.flowables import Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import os
import mysql.connector

# File path for the CSV file (same directory as this script)
file_path = os.path.join(os.path.dirname(__file__), 'mlb_2024.csv')

# Replace these placeholders with your actual database credentials
db_config = {
    'user': 'INPUT USERNAME',  # Your MySQL username
    'password': 'INPUT PASSWORD',  # Your MySQL password
    'host': 'INPUT HOST',  # Typically 'localhost' for local setups
    'database': 'INPUT DATABASE'  # The name of your database
}

# Function to connect to MySQL database
def connect_to_database():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    
def get_player_id_by_name(player_name):
    """
    Retrieve player ID from the MySQL database based on player name.
    """
    try:
        connection = connect_to_database()
        cursor = connection.cursor()
        
        # Use a case-insensitive search to find the player
        query = "SELECT player_id FROM players WHERE LOWER(name) = LOWER(%s)"
        cursor.execute(query, (player_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Function to look up player name by ID from MySQL database
def get_player_name(player_id):
    connection = connect_to_database()
    if connection:
        cursor = connection.cursor()
        query = "SELECT name FROM players WHERE player_id = %s"
        cursor.execute(query, (player_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result:
            return result[0]
        else:
            return None
    else:
        return None

# Pitch type mapping for readable names
pitch_type_names = {
    'CH': 'Changeup', 'CU': 'Curveball', 'FC': 'Cutter', 'EP': 'Eephus', 'FO': 'Forkball',
    'FF': '4S FB', 'KN': 'Knuckleball', 'KC': 'Knuckle-curve', 'SC': 'Screwball',
    'SI': 'Sinker', 'SL': 'Slider', 'SV': 'Slurve', 'FS': 'Splitter', 'ST': 'Sweeper'
}

def load_and_filter_data(file_path, batter_id=None, pitcher_id=None):
    """
    Load and filter data based on the selected pitcher and batter IDs.
    Returns a DataFrame for analyzing the input pitcher facing batters of the correct side,
    and details about pitcher and batter.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)

        # Map pitch types to readable names
        if 'pitch_type' in df.columns:
            df['pitch_type'] = df['pitch_type'].map(pitch_type_names).fillna(df['pitch_type'])

        # Use default values if inputs are empty or invalid
        if not batter_id or batter_id < 1:
            batter_id = 467793  # Default batter ID
        if not pitcher_id or pitcher_id < 1:
            pitcher_id = 666142  # Default pitcher ID

        # Get player names from the database
        pitcher_name = get_player_name(pitcher_id)
        if pitcher_name is None:
            pitcher_name = f'Pitcher {pitcher_id}'

        batter_name = get_player_name(batter_id)
        if batter_name is None:
            batter_name = f'Batter {batter_id}'

        # Extract pitch hand for the pitcher
        pitch_hand = df.loc[df['pitcher_id'] == pitcher_id, 'pitch_hand'].iloc[0]

        # Filter to only rows for the specified pitcher_id
        pitcher_df = df[df['pitcher_id'] == pitcher_id]

        # If the pitcher has no rows, return empty DataFrame
        if pitcher_df.empty:
            print(f"No data found for pitcher_id {pitcher_id}.")
            return pd.DataFrame(), pd.DataFrame(), pitcher_name, batter_name, pitch_hand, None

        # Determine the batter's hitting side(s)
        batter_sides = df.loc[df['batter_id'] == batter_id, 'bat_side'].unique()
        if len(batter_sides) > 1:  # Switch hitter
            # Switch hitters face the opposite side of the pitcher's throwing hand
            batter_side = "L" if pitch_hand == "R" else "R"
            filtered_pitcher_df = pitcher_df[pitcher_df['bat_side'] != pitch_hand]
        else:
            batter_side = batter_sides[0]
            filtered_pitcher_df = pitcher_df[pitcher_df['bat_side'] == batter_side]

        # Filter out pitch types with no data for the selected batter side
        pitch_types_with_data = filtered_pitcher_df['pitch_type'].unique()
        filtered_pitcher_df = filtered_pitcher_df[filtered_pitcher_df['pitch_type'].isin(pitch_types_with_data)]

        # Filter batter data for pitchers with the same pitch_hand as the input pitcher
        filtered_batter_df = df[
            (df['batter_id'] == batter_id) &
            (df['pitch_hand'] == pitch_hand) &
            (df['pitch_type'].isin(pitch_types_with_data))
        ]

        # Check if the filtered pitcher data contains at least one pitch for the selected batter side
        if filtered_pitcher_df.empty:
            print(f"No data found for pitcher_id {pitcher_id} with pitches thrown to {batter_side} side.")
            return pd.DataFrame(), pd.DataFrame(), pitcher_name, batter_name, pitch_hand, batter_side

        return filtered_pitcher_df, filtered_batter_df, pitcher_name, batter_name, pitch_hand, batter_side

    except FileNotFoundError:
        print("Error: File not found. Check the file path.")
    except KeyError as e:
        print(f"Error: Missing column in the CSV file - {e}")

def draw_strike_zone(ax):
    """
    Draws the strike zone on the provided Axes object.
    """
    # Define the strike zone rectangle
    strike_zone = patches.Rectangle((-0.83, 1.5), 1.66, 2.0, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(strike_zone)

    # Add dashed grid lines for better visualization
    for i in range(1, 3):
        x = -0.83 + i * (1.66 / 3)  # Horizontal divisions
        y = 1.5 + i * (2.0 / 3)     # Vertical divisions
        ax.plot([x, x], [1.5, 3.5], color='gray', linestyle='--', linewidth=1)  # Vertical dashed lines
        ax.plot([-0.83, 0.83], [y, y], color='gray', linestyle='--', linewidth=1)  # Horizontal dashed lines

    return ax

def generate_heatmap_images(filtered_pitcher_df, batter_side, pitcher_id, output_dir="temp_heatmaps"):
    """
    Generate heatmap images for each pitch type, showing pitch locations relative to the strike zone.
    Parameters:
        filtered_pitcher_df (pd.DataFrame): Filtered data for the pitcher.
        batter_side (str): Batter's hitting side ("R", "L", "Switch").
        pitcher_id (int): Pitcher's ID number, included in the image filenames.
        output_dir (str): Directory to save the heatmap images.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each unique pitch type
    image_paths = []
    for pitch_type in filtered_pitcher_df['pitch_type'].unique():
        pitch_data = filtered_pitcher_df[filtered_pitcher_df['pitch_type'] == pitch_type]

        # Skip if no data for the pitch type or not enough data points
        if pitch_data.empty or len(pitch_data) < 5:  # Set a threshold, e.g., 5 pitches minimum
            continue

        # Create the heatmap figure
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax = draw_strike_zone(ax)

        # Generate the KDE heatmap
        try:
            sns.kdeplot(
                x=pitch_data['plate_x'], 
                y=pitch_data['plate_z'], 
                fill=True, cmap='YlOrRd', ax=ax, 
                bw_adjust=0.7, alpha=0.6
            )
        except ValueError:
            # If KDE fails due to singular data, fallback to scatter plot
            sns.scatterplot(
                x=pitch_data['plate_x'],
                y=pitch_data['plate_z'],
                color='red',
                ax=ax
            )

        # Set heatmap attributes
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        ax.set_xlabel("Horizontal Location", fontsize=12, fontweight='bold')
        ax.set_ylabel("Vertical Location", fontsize=12, fontweight='bold')

        # Adjust title logic for switch hitters
        if batter_side == "Switch":
            pitcher_hand = filtered_pitcher_df['pitch_hand'].iloc[0]
            opposite_side = "LHH" if pitcher_hand == "R" else "RHH"
            ax.set_title(f"{pitch_type} vs. {opposite_side}", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{pitch_type} vs. {batter_side}HH", fontsize=14, fontweight='bold')

        ax.set_aspect('equal', 'box')

        # Generate the filename
        filename = f"{pitcher_id}_{pitch_type}_vs_{batter_side}HH_heatmap.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        image_paths.append(output_path)  # Store path for later use

        plt.close(fig)  # Close the figure to avoid memory issues

    return image_paths

def analyze_game_outcomes(filtered_pitcher_df, filtered_batter_df):
    """
    Analyze game outcomes for a given pitcher and batter matchup.
    Combines pitch summaries, outcome percentages, and batter summaries.
    """
    # Define groupings
    swing_outcomes = ["Swinging Strike", "In play, no out", "Foul", "In play, out(s)", 
                      "In play, run(s)", "Foul Tip", "Swinging Strike (Blocked)", "Foul Bunt", "Missed Bunt"]
    whiff_outcomes = ["Swinging Strike", "Swinging Strike (Blocked)", "Missed Bunt"]
    ball_outcomes = ["Ball", "Ball In Dirt", "Hit By Pitch"]
    strike_outcomes = swing_outcomes + ["Called Strike"]

    # Map pitch types to readable names for both DataFrames
    if 'pitch_type' in filtered_pitcher_df.columns:
        filtered_pitcher_df['pitch_type'] = filtered_pitcher_df['pitch_type'].map(pitch_type_names).fillna(filtered_pitcher_df['pitch_type'])

    if 'pitch_type' in filtered_batter_df.columns:
        filtered_batter_df['pitch_type'] = filtered_batter_df['pitch_type'].map(pitch_type_names).fillna(filtered_batter_df['pitch_type'])

    # Add Outcome Type column to both DataFrames
    filtered_pitcher_df['Outcome Type'] = filtered_pitcher_df['description'].apply(
        lambda x: "Ball" if x in ball_outcomes else ("Strike" if x in strike_outcomes else "Other")
    )

    filtered_batter_df['Outcome Type'] = filtered_batter_df['description'].apply(
        lambda x: "Ball" if x in ball_outcomes else ("Strike" if x in strike_outcomes else "Other")
    )

    # Add Play Outcome column
    def classify_description(row, next_row):
        description = row['description']
        current_event_index = row['event_index']
        next_event_index = next_row['event_index'] if next_row is not None else None

        # Conditional logic for play outcomes
        if description == "Ball":
            return ("Walk", "On Base") if current_event_index != next_event_index else ("Ball", "Ongoing AB")
        elif description == "Swinging Strike":
            return ("Strikeout", "Out") if current_event_index != next_event_index else ("Swinging Strike", "Ongoing AB")
        elif description == "In play, no out":
            return "Hit", "On Base"
        elif description == "Called Strike":
            return ("Strikeout", "Out") if current_event_index != next_event_index else ("Called Strike", "Ongoing AB")
        elif description == "Foul":
            return "Foul", "Ongoing AB"
        elif description == "In play, out(s)":
            return "Out", "Out"
        elif description == "Ball In Dirt":
            return ("Walk", "On Base") if current_event_index != next_event_index else ("Ball In Dirt", "Ongoing AB")
        elif description == "In play, run(s)":
            return "In play, run(s)", "On Base"
        elif description == "Foul Tip":
            return ("Strikeout", "Out") if current_event_index != next_event_index else ("Foul Tip", "Ongoing AB")
        elif description == "Swinging Strike (Blocked)":
            return ("Strikeout", "Out") if current_event_index != next_event_index else ("Swinging Strike (Blocked)", "Ongoing AB")
        elif description == "Foul Bunt":
            return ("Foul Bunt", "Out") if current_event_index != next_event_index else ("Foul Bunt", "Ongoing AB")
        elif description == "Hit By Pitch":
            return "Hit By Pitch", "On Base"
        elif description == "Missed Bunt":
            return ("Strikeout", "Out") if current_event_index != next_event_index else ("Missed Bunt", "Ongoing AB")
        else:
            return "Unknown", "Other"

    # Function to classify outcomes for a given DataFrame
    def classify_outcomes(df):
        outcomes, outcome_groups = [], []
        for idx in range(len(df)):
            row = df.iloc[idx]
            next_row = df.iloc[idx + 1] if idx + 1 < len(df) else None
            outcome, outcome_group = classify_description(row, next_row)
            outcomes.append(outcome)
            outcome_groups.append(outcome_group)
        df['Outcome'] = outcomes
        df['Play Outcome'] = outcome_groups
        return df

    # Apply classification to both DataFrames
    filtered_pitcher_df = classify_outcomes(filtered_pitcher_df)
    filtered_batter_df = classify_outcomes(filtered_batter_df)

    # Embedded function to calculate Whiff %
    def calculate_whiff_percentage(data):
        """
        Calculate the Whiff % for each pitch type.
        """
        whiff_summary = []
        for pitch_type, group in data.groupby('pitch_type'):
            swings = group[group['description'].isin(swing_outcomes)]
            whiffs = swings[swings['description'].isin(whiff_outcomes)]
            whiff_percentage = (len(whiffs) / len(swings) * 100) if len(swings) > 0 else 0.0
            whiff_summary.append({"pitch_type": pitch_type, "Whiff %": whiff_percentage})
        return pd.DataFrame(whiff_summary).set_index('pitch_type')

    def create_combined_pitch_summary(filtered_pitcher_df):
        """
        Create a combined summary of pitch metrics and outcome percentages.
        Handles non-numeric values and missing data.
        """
        # Clean non-numeric values in numeric columns
        numeric_columns = ['release_speed', 'horz_break', 'induced_vert_break']
        for col in numeric_columns:
            filtered_pitcher_df[col] = pd.to_numeric(filtered_pitcher_df[col], errors='coerce')  # Coerce invalid entries to NaN

        # Calculate total pitches
        total_pitches = len(filtered_pitcher_df)

        # Group and calculate summary statistics for pitch metrics
        summary = (
            filtered_pitcher_df.groupby('pitch_type')
            .agg(
                pitch_count=('pitch_type', 'size'),
                average_horz_break=('horz_break', lambda x: -x.mean(skipna=True)),
                average_induced_vert_break=('induced_vert_break', lambda x: x.mean(skipna=True)),
                average_velocity=('release_speed', lambda x: x.mean(skipna=True)),
                max_velocity=('release_speed', lambda x: x.max(skipna=True))
            )
            .reset_index()
        )

        # Add usage percentage as float
        summary['Usage %'] = (summary['pitch_count'] / total_pitches * 100).round(1)

        # Calculate outcome percentages
        relevant_data = filtered_pitcher_df[filtered_pitcher_df['Play Outcome'].isin(['Out', 'On Base'])]
        strike_counts = filtered_pitcher_df[filtered_pitcher_df['Outcome Type'] == 'Strike'].groupby('pitch_type').size()
        swing_counts = filtered_pitcher_df[filtered_pitcher_df['description'].isin(swing_outcomes)].groupby('pitch_type').size()
        total_counts = filtered_pitcher_df['pitch_type'].value_counts()
        strikeout_counts = relevant_data[relevant_data['Outcome'] == 'Strikeout'].groupby('pitch_type').size()
        walk_counts = relevant_data[relevant_data['Outcome'] == 'Walk'].groupby('pitch_type').size()
        out_counts = relevant_data[relevant_data['Play Outcome'] == 'Out'].groupby('pitch_type').size()
        on_base_counts = relevant_data[relevant_data['Play Outcome'] == 'On Base'].groupby('pitch_type').size()

        # Calculate percentages and metrics
        percentages = []
        whiff_df = calculate_whiff_percentage(filtered_pitcher_df)

        for pitch_type in total_counts.index:
            total_outcomes = out_counts.get(pitch_type, 0) + on_base_counts.get(pitch_type, 0)
            percentages.append({
                "pitch_type": pitch_type,
                "Strike %": (strike_counts.get(pitch_type, 0) / total_counts[pitch_type] * 100) if total_counts[pitch_type] > 0 else 0.0,
                "Swing %": (swing_counts.get(pitch_type, 0) / total_counts[pitch_type] * 100) if total_counts[pitch_type] > 0 else 0.0,
                "Strikeout %": (strikeout_counts.get(pitch_type, 0) / total_outcomes * 100) if total_outcomes > 0 else 0.0,
                "Walk %": (walk_counts.get(pitch_type, 0) / total_outcomes * 100) if total_outcomes > 0 else 0.0,
                "Whiff %": whiff_df.loc[pitch_type, "Whiff %"] if pitch_type in whiff_df.index else 0.0
            })

        # Convert percentages to DataFrame
        percentages_df = pd.DataFrame(percentages)

        # Merge pitch metrics and percentages
        combined_pitch_summary = summary.merge(percentages_df, on='pitch_type', how='left')

        # Rename columns for readability
        combined_pitch_summary = combined_pitch_summary.rename(columns={
            'pitch_type': 'Pitch Type',
            'average_horz_break': 'Average HB',
            'average_induced_vert_break': 'Average IVB',
            'average_velocity': 'Average Velocity',
            'max_velocity': 'Max Velocity',
            'pitch_count': '# Pitches',
            'Strike %': 'Strike %',
            'Swing %': 'Swing %',
            'Strikeout %': 'Strikeout %',
            'Walk %': 'Walk %',
            'Whiff %': 'Whiff %'
        }).round(1)

        #filter less than 1% usage %
        combined_pitch_summary = combined_pitch_summary[combined_pitch_summary['Usage %'] >= 1]

        # Sort by "Usage %" in descending order
        combined_pitch_summary = combined_pitch_summary.sort_values(by='Usage %', ascending=False)

        # Reorder columns for output
        combined_pitch_summary = combined_pitch_summary[
            ['Pitch Type', 'Usage %', 'Max Velocity', 'Average Velocity', 'Average IVB',
            'Average HB', 'Strike %', 'Swing %', 'Whiff %', 'Strikeout %', 'Walk %']
        ]

        # Reset index for cleaner display
        combined_pitch_summary.reset_index(drop=True, inplace=True)

        return combined_pitch_summary

    def create_combined_batter_summary(filtered_batter_df, filtered_pitcher_df):
        """
        Create a combined summary of batter performance against pitch types.
        Filters and calculates metrics for the batter's performance against all pitchers
        with the same pitch_hand as the input pitcher_id. Excludes pitch types with Usage % <= 1% in the pitcher table.
        """

        # Calculate pitch usage percentage for the pitcher
        total_pitches = len(filtered_pitcher_df)
        pitcher_usage = (
            filtered_pitcher_df.groupby('pitch_type')
            .size()
            .reset_index(name='pitch_count')
        )
        pitcher_usage['Usage %'] = (pitcher_usage['pitch_count'] / total_pitches * 100).round(1)

        # Get a list of pitch types with Usage % > 1 in the pitcher table
        relevant_pitch_types = pitcher_usage[pitcher_usage['Usage %'] > 1]['pitch_type'].tolist()

        # Filter the batter dataframe to include only relevant pitch types
        filtered_batter_df = filtered_batter_df[filtered_batter_df['pitch_type'].isin(relevant_pitch_types)]

        # Group and calculate summary statistics for batter metrics
        batter_summary = (
            filtered_batter_df.groupby('pitch_type')
            .agg(
                # Count total appearances (# Seen) for each pitch type
                seen=('pitch_type', 'size'),

                # Average Exit Velocity (ignores NA values)
                average_launch_speed=('launch_speed', lambda x: x.dropna().mean()),

                # Hard Hit %: Count of launch_speed >= 95 divided by valid entries
                hard_hit_percent=('launch_speed', lambda x: ((x.dropna() >= 95).sum() / len(x.dropna())) * 100 if len(x.dropna()) > 0 else 0.0),

                # Whiff %: Use swings and whiffs for calculation
                whiff_percent=('description', lambda x: (
                    (x.isin(whiff_outcomes).sum() / x.isin(swing_outcomes).sum()) * 100
                    if x.isin(swing_outcomes).sum() > 0 else 0.0
                ))
            )
            .reset_index()
        )

        # Replace NaN values with 0.0 for numeric columns
        batter_summary = batter_summary.fillna({'average_launch_speed': 0.0, 'hard_hit_percent': 0.0})

        # Rename columns for readability
        batter_summary = batter_summary.rename(columns={
            'pitch_type': 'Pitch Type',
            'seen': '# Seen',
            'average_launch_speed': 'Avg EV',
            'hard_hit_percent': 'Hard Hit %',
            'whiff_percent': 'Whiff %'
        }).round(1)

        # Sort by "# Seen" in descending order
        batter_summary = batter_summary.sort_values(by='# Seen', ascending=False)

        # Reorder columns for display
        batter_summary = batter_summary[['Pitch Type', '# Seen', 'Avg EV', 'Hard Hit %', 'Whiff %']]

        # Reset index for cleaner display
        batter_summary.reset_index(drop=True, inplace=True)

        return batter_summary

    final_pitch_summary = create_combined_pitch_summary(filtered_pitcher_df)
    final_batter_summary = create_combined_batter_summary(filtered_batter_df, filtered_pitcher_df)
    
    # return final_pitch_summary, final_batter_summary
    return final_pitch_summary, final_batter_summary

def clean_table_units(df, columns_with_units):
    df_cleaned = df.copy()
    for col in columns_with_units:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str)
                                            .str.replace(' mph', '', regex=False)
                                            .str.replace(' %', '', regex=False)
                                            .str.replace(' in', '', regex=False), 
                                            errors='coerce')
    return df_cleaned
    
def create_notes_image(notes, output_path="notes_section.png"):
    """
    Creates an image of the notes section with the given content using matplotlib.
    """
    # Create a wider figure to expand horizontally
    fig, ax = plt.subplots(figsize=(6, 2))  # Width is increased
    ax.axis('off')  # Turn off the axes

    # Static title
    title = "Key Notes"
    ax.text(0.05, 0.9, title, fontsize=11, fontweight='bold', ha='left', va='top')  # Left-aligned

    # Add space between the title and the bullet points
    start_y = 0.7  # Adjust this value to increase or decrease the spacing below the title

    # Add the notes dynamically
    for i, note in enumerate(notes):
        ax.text(0.05, start_y - i * 0.18, f"• {note}", fontsize=10, ha='left', va='top')

    # Save the figure as a PNG file
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def generate_reportlab_pdf(pitcher_name, batter_name, pitch_hand, batter_side, final_pitch_summary, final_batter_summary, filtered_pitcher_df):
    """
    Generate a PDF with dynamic placement of a notes image.
    """
    # Define output paths
    output_pdf_file = "hitting_report_with_notes.pdf"
    temp_pdf_file = "temp_report.pdf"
    notes_image_path = "notes_section.png"

    # Generate notes dynamically
    pitch_summary_cleaned = clean_table_units(final_pitch_summary, ['Max Velocity', 'Avg Velocity', 'Strike %', 'Whiff %', 'Strikeout %', 'Walk %'])
    batter_summary_cleaned = clean_table_units(final_batter_summary, ['Avg EV', 'Hard Hit %', 'Whiff %'])
    
    key_notes = []

    # Variables and lists to build, notes to follow in specific sequence
    fastball_variants = ['4S FB', 'Sinker']
    
    max_velocity_high = pitch_summary_cleaned['Max Velocity'].max()
    avg_velocity_high = pitch_summary_cleaned['Average Velocity'].max()
    max_velocity_low = pitch_summary_cleaned['Max Velocity'].max()
    avg_velocity_low = pitch_summary_cleaned['Average Velocity'].max()
    max_usage_pitch = pitch_summary_cleaned['Usage %'].max()
    pitch_favored = pitch_summary_cleaned[pitch_summary_cleaned['Usage %'] > 50]
    
    pitch_summary_cleaned['Weighted Strike Contribution'] = (
        pitch_summary_cleaned['Usage %'] * pitch_summary_cleaned['Strike %'] / 100
    )
    weighted_strike_percentage = pitch_summary_cleaned['Weighted Strike Contribution'].sum()

    high_whiff_pitch_types = []
    low_whiff_pitch_types = []
    strikeout_pitches = []
    dead_zone_pitches = []

    # Bad Matchup note
    for _, row in batter_summary_cleaned.iterrows():
        pitch_type = row['Pitch Type']
        batter_whiff = row['Whiff %']
        batter_seen = row['# Seen']
        # Check if batter's Whiff % is greater than 30% and # Seen is greater than 30
        if batter_whiff > 30 and batter_seen > 30:
            # Check if the same pitch type exists in the pitcher table with Whiff % > 30%
            pitcher_row = pitch_summary_cleaned[pitch_summary_cleaned['Pitch Type'] == pitch_type]
            if not pitcher_row.empty and pitcher_row['Whiff %'].iloc[0] > 30:
                high_whiff_pitch_types.append(pitch_type)
    if high_whiff_pitch_types:
        key_notes.append(f"Bad matchup vs. {', '.join(high_whiff_pitch_types)} - see the action and defend")
        
    # Good Matchup note
    for _, row in batter_summary_cleaned.iterrows():
        pitch_type = row['Pitch Type']
        batter_whiff = row['Whiff %']
        batter_seen = row['# Seen']
        # Check if batter's Whiff % is greater than 30% and # Seen is greater than 30
        if batter_whiff < 22 and batter_seen > 30:
            # Check if the same pitch type exists in the pitcher table with Whiff % > 30%
            pitcher_row = pitch_summary_cleaned[pitch_summary_cleaned['Pitch Type'] == pitch_type]
            if not pitcher_row.empty and pitcher_row['Whiff %'].iloc[0] < 22:
                low_whiff_pitch_types.append(pitch_type)
    if low_whiff_pitch_types:
        key_notes.append(f"Good matchup vs. {', '.join(low_whiff_pitch_types)} - see for strikes and attack")

    # Dead zone fastball note for 4S FB and Sinker
    for pitch_type in fastball_variants:
        pitch_row = pitch_summary_cleaned[pitch_summary_cleaned['Pitch Type'] == pitch_type]
        if not pitch_row.empty:
            # Check if Usage % is at least 10%
            usage = pitch_row['Usage %'].iloc[0]
            if usage >= 10:
                hb = pitch_row['Average HB'].iloc[0]
                ivb = pitch_row['Average IVB'].iloc[0]
                if (8 <= abs(hb) <= 13) and (8 <= ivb <= 13):
                    dead_zone_pitches.append(pitch_type)
    if dead_zone_pitches:
        key_notes.append(f"{pitcher_name} throws a dead zone {', '.join(dead_zone_pitches)}")

    # Pitcher Strikeout Note
    for _, row in pitch_summary_cleaned.iterrows():
        pitch_type = row['Pitch Type']
        pitcher_strikeout = row['Strikeout %']
        usage = row['Usage %']  
        if pitcher_strikeout > 30 and usage > 10:
            strikeout_pitches.append(pitch_type)
    if strikeout_pitches:
        key_notes.append(f"{', '.join(strikeout_pitches)} are {pitcher_name}'s strikeout pitches")
       
    # Note for no pitch favored
    if max_usage_pitch <= 35:
        key_notes.append(f"{pitcher_name} does not favor any one pitch")

    # Note for pitch favored over 50%
    if not pitch_favored.empty:
        favored_pitch = pitch_favored.iloc[0]  # Get the first row of the filtered DataFrame
        favored_pitch_type = favored_pitch['Pitch Type']
        favored_strike_percent = favored_pitch['Strike %']
        key_notes.append(f"{pitcher_name} favors {favored_pitch_type} over half of the time with {favored_strike_percent} % strikes")

    #IVB and HB checks on fastball
    for pitch_type in fastball_variants:
        pitch_row = pitch_summary_cleaned[pitch_summary_cleaned['Pitch Type'] == pitch_type]
        if not pitch_row.empty:
            # Check if Usage % is at least 10%
            usage = pitch_row['Usage %'].iloc[0]
            if usage >= 10:
                # Check for breaks
                ivb = pitch_row['Average IVB'].iloc[0]
                hb = pitch_row['Average HB'].iloc[0]
                # Check ivb conditions
                if ivb >= 16 and abs(hb) >= 16:
                    key_notes.append(f"{pitcher_name} has a {pitch_type} that will appear to rise and run arm side")
                elif ivb >= 16:
                    key_notes.append(f"{pitcher_name} has a straight {pitch_type} that will appear to rise")
                # Check hb conditions
                if abs(hb) >= 16 and ivb <= 6:
                    key_notes.append(f"{pitcher_name} has a {pitch_type} that will move heavy and run arm side")
                elif abs(hb) >= 16:
                    key_notes.append(f"{pitcher_name} has a flat {pitch_type} with a lot of arm side run")

    # strike percentage note
    if weighted_strike_percentage > 60:
        key_notes.append(f"{pitcher_name} is a high strike pitcher, be ready to attack")
    elif weighted_strike_percentage < 55:
        key_notes.append(f"{pitcher_name} is a low strike pitcher, be ready to see pitches")

    # High velocity note
    if max_velocity_high >= 100 and avg_velocity_high >= 97:
        key_notes.append(f"{pitcher_name} has elite velocity stamina and peak potential")
    elif max_velocity_high >= 100:
        key_notes.append(f"{pitcher_name} has a high peak velocity, above 100 mph")
    elif avg_velocity_high >= 97:
        key_notes.append(f"{pitcher_name} has a high average velocity, above 97 mph")

    # Low velocity note
    if max_velocity_low <= 92 and avg_velocity_low <= 89:
        key_notes.append(f"{pitcher_name} is a low velocity pitcher")
    elif max_velocity_low <= 92:
        key_notes.append(f"{pitcher_name} has a slow max velocity")
    elif avg_velocity_low <= 89:
        key_notes.append(f"{pitcher_name} has a slow average velocity")

    # Average velocity note for 4S FB and Sinker
    for pitch_type in fastball_variants:
        pitch_row = pitch_summary_cleaned[pitch_summary_cleaned['Pitch Type'] == pitch_type]
        if not pitch_row.empty:
            avg_velocity = pitch_row['Average Velocity'].iloc[0]
            note = f"{pitcher_name} has average velocity"  # Define the note
            if 90 <= avg_velocity <= 93 and note not in key_notes:
                key_notes.append(note)

    key_notes = key_notes[:5]  # Limit to 5 notes
    create_notes_image(key_notes, notes_image_path)  
      
    # Generate heatmaps and get their paths
    heatmap_image_paths = generate_heatmap_images(filtered_pitcher_df, batter_side, pitcher_id)

    # Create document
    output_pdf_file = 'hitting_report.pdf'
    doc = SimpleDocTemplate(temp_pdf_file, pagesize=landscape(letter),
                            leftMargin=0.2 * inch, rightMargin=0.2 * inch,
                            topMargin=0.2 * inch, bottomMargin=0.1 * inch)

    elements = []

    # Title and author with adjusted font sizes
    title_style = ParagraphStyle(name="Title", fontSize=18, alignment=TA_CENTER, fontName="Helvetica-Bold")
    author_style = ParagraphStyle(name="Author", fontSize=10, alignment=TA_CENTER, fontName="Helvetica-Bold")
    elements.append(Paragraph(f"Hitting Report - {batter_name} vs. {pitch_hand}HP {pitcher_name}", title_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Report by Robbie Dudzinski", author_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Update pitcher summary headings
    final_pitch_summary = final_pitch_summary.rename(columns={
        'Average Velocity': 'Avg Velocity',
        'Average IVB': 'Avg IVB',
        'Average HB': 'Avg HB'
    })

    # Formatting the final_pitch_summary
    final_pitch_summary['Max Velocity'] = final_pitch_summary['Max Velocity'].astype(str) + " mph"
    final_pitch_summary['Avg Velocity'] = final_pitch_summary['Avg Velocity'].astype(str) + " mph"
    final_pitch_summary['Avg IVB'] = final_pitch_summary['Avg IVB'].astype(str) + " in"
    final_pitch_summary['Avg HB'] = final_pitch_summary['Avg HB'].astype(str) + " in"
    percentage_columns = ['Usage %', 'Strike %', 'Swing %', 'Whiff %', 'Strikeout %', 'Walk %']
    for col in percentage_columns:
        final_pitch_summary[col] = final_pitch_summary[col].astype(str) + " %"

    # Pitcher report table with adjusted column widths to span the entire page width
    subtitle_style = ParagraphStyle(name="Subtitle", fontSize=12, alignment=TA_LEFT, fontName="Helvetica-Bold")
    elements.append(Paragraph(f"{pitcher_name} Pitch Data vs. {batter_side}HH", subtitle_style))
    elements.append(Spacer(1, 0.1 * inch))

    num_columns = len(final_pitch_summary.columns)
    pitch_data = [list(final_pitch_summary.columns)] + final_pitch_summary.values.tolist()
    pitch_table = Table(pitch_data, colWidths=[(10.5 * inch) / num_columns] * num_columns)

    # Define velocity thresholds for each pitch type by pitcher hand
    avg_velocity_thresholds = {
        '4S FB': {'R': 92.4, 'L': 91.2},
        'Sinker': {'R': 91.9, 'L': 90.8},
        'Cutter': {'R': 88.6, 'L': 86.2},
        'Slider': {'R': 84.3, 'L': 82.9},
        'Curveball': {'R': 77.8, 'L': 76.0},
        'Slurve': {'R': 77.8, 'L': 76.0},
        'Screwball': {'R': 77.8, 'L': 76.0},
        'Knuckle-curve': {'R': 77.8, 'L': 76.0},
        'Sweeper': {'R': 77.8, 'L': 76.0},
        'Changeup': {'R': 83.7, 'L': 82.1},
        'Splitter': {'R': 83.7, 'L': 82.1},
        'Forkball': {'R': 83.7, 'L': 82.1},
    }
    
    # Define usage thresholds for each pitch type
    usage_thresholds = {
        '4S FB': 51.1,
        'Sinker': 51.1,
        'Slider': 18.3,
        'Changeup': 10.6,
        'Curveball': 9.5,
        'Cutter': 5.2,
        'Splitter': 1.3,
        # All other pitch types fall under "Other"
    }
    
    # Define movement thresholds for each pitch type
    movement_thresholds = {
    '4S FB': {'IVB': {'min': 14, 'max': 30}, 'HB': {'highlight': 'green', 'min': 13}},
    'Sinker': {'IVB': {'min': -10, 'max': 10}, 'HB': {'min': 13, 'max': 30}},
    'Cutter': {'IVB': {'min': -10, 'max': 10}, 'HB': {'min': 6, 'max': 6}},
    'Slider': {'IVB': {'min': -6, 'max': 3}, 'HB': {'min': 4, 'max': 30}},
    'Curveball': {'IVB': {'min': -30, 'max': -14}, 'HB': {'min': 1, 'max': 10}},
    'Slurve': {'IVB': {'min': -20, 'max': -10}, 'HB': {'min': 10, 'max': 30}},
    'Knuckle-curve': {'IVB': {'min': -30, 'max': -14}, 'HB': {'min': 1, 'max': 10}},
    'Changeup': {'IVB': {'min': -10, 'max': 9}, 'HB': {'min': 10, 'max': 30}},
    'Splitter': {'IVB': {'min': -10, 'max': 9}, 'HB': {'min': 5, 'max': 30}},
    'Forkball': {'IVB': {'min': -10, 'max': 9}, 'HB': {'min': 5, 'max': 30}},
    # Add more pitch types if necessary, or provide default thresholds for other pitches
}

    # Define the pitcher table style with conditional formatting
    pitch_table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),  # Bold first column (Pitch Type)
    ]

    # Apply conditional formatting for pitcher table
    for row_idx, row in enumerate(final_pitch_summary.values, start=1):
        for col_idx, value in enumerate(row):
            # Remove the units for comparison
            if isinstance(value, str) and ('mph' in value or 'in' in value or '%' in value):
                value = value.split()[0]

            try:
                value = float(value)
            except ValueError:
                continue

            # Get the pitch type for this row
            pitch_type = row[0]  # Assuming the first column contains the pitch type

            # Usage % logic
            if col_idx == 1:  # 'Usage %'
                # Determine the threshold for the current pitch type
                usage_threshold = usage_thresholds.get(pitch_type, 1.0)  # Default to 1% for "Other" pitches

                # Apply green if above the threshold, pink if below
                if value >= usage_threshold:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                else:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Get the pitch type name
            pitch_type_name = row[0]  # Assuming the first column contains the pitch type name

            # Max Velocity logic
            if col_idx == 2:  # 'Max Velocity'
                # Get the average velocity threshold for the pitch type and pitcher hand
                if pitch_type_name in avg_velocity_thresholds:
                    velocity_threshold = avg_velocity_thresholds[pitch_type_name].get(pitch_hand, None)

                    if velocity_threshold is not None:
                        max_velocity_threshold = velocity_threshold + 3  # Max velocity threshold is 3 higher than avg
                        if value >= max_velocity_threshold:
                            pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                        else:
                            pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Average Velocity logic
            if col_idx == 3:  # 'Avg Velocity'
                # Get the average velocity threshold for the pitch type and pitcher hand
                if pitch_type_name in avg_velocity_thresholds:
                    velocity_threshold = avg_velocity_thresholds[pitch_type_name].get(pitch_hand, None)

                    if velocity_threshold is not None:
                        if value > velocity_threshold:
                            pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                        elif value < velocity_threshold:
                            pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Avg IVB logic
            if col_idx == 4 and pitch_type_name in movement_thresholds:  # 'Avg IVB'
                ivb_threshold = movement_thresholds[pitch_type_name]['IVB']
                if ivb_threshold['min'] <= value <= ivb_threshold['max']:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                else:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))


            # Avg HB logic
            if col_idx == 5 and pitch_type_name in movement_thresholds:  # 'Avg HB'
                hb_threshold = movement_thresholds[pitch_type_name]['HB']
                if pitch_type_name == '4S FB' and hb_threshold.get('highlight') == 'green':
                    # Special condition for '4S FB' to only highlight green if HB >= 13
                    if value >= hb_threshold['min']:
                        pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                    # No 'else' here means no other color is applied (no pink).
                else:
                    # Default logic for other pitch types using abs(value)
                    if hb_threshold['min'] <= abs(value) <= hb_threshold['max']:
                        pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                    else:
                        pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))             
        
            # Strike % logic
            if col_idx == 6:  # 'Strike %'
                if value >= 63:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value <= 56:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Swing % logic
            if col_idx == 7:  # 'Swing %'
                if value >= 50:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value <= 40:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Whiff % logic
            if col_idx == 8:  # 'Whiff %'
                if value >= 30:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value <= 22:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Strikeout % logic
            if col_idx == 9:  # 'Strikeout %'
                if value >= 30:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value <= 15:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

            # Walk % logic
            if col_idx == 10:  # 'Walk %'
                if value <= 5.5:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value >= 11:
                    pitch_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))

    pitch_table.setStyle(TableStyle(pitch_table_style))
    pitch_table.hAlign = 'LEFT'
    elements.append(pitch_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Subtitle for heatmaps
    elements.append(Paragraph(f"{pitcher_name} Pitch Location Heatmaps vs. {batter_side}HH", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Add heatmap images in one row
    num_images = len(heatmap_image_paths)
    if num_images > 0:
        # Set a maximum width for each image
        max_image_width = 2 * inch  
        
        # Calculate the appropriate width for each image
        image_width = min((10 * inch) / num_images, max_image_width)  # Use min to constrain the width
        image_height = image_width  # Keep aspect ratio square for consistency

        heatmap_images = [Image(img_path, width=image_width, height=image_height) for img_path in heatmap_image_paths]
        heatmap_table = Table([heatmap_images], colWidths=[image_width] * num_images)
        heatmap_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ]))
        elements.append(heatmap_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Formatting the final_batter_summary
    final_batter_summary['Avg EV'] = final_batter_summary['Avg EV'].astype(str) + " mph"
    percentage_columns_batter = ['Hard Hit %', 'Whiff %']
    for col in percentage_columns_batter:
        final_batter_summary[col] = final_batter_summary[col].astype(str) + " %"

    # Batter report table
    elements.append(Paragraph(f"{batter_name} Splits vs. {pitch_hand}HP", subtitle_style))
    elements.append(Spacer(1, 0.1 * inch))

    batter_data = [list(final_batter_summary.columns)] + final_batter_summary.values.tolist()
    batter_table = Table(batter_data, colWidths=[1.5 * inch] + [0.75 * inch] * (len(final_batter_summary.columns) - 1))

    # Define the batter table style with conditional formatting
    batter_table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),  # Bold first column (Pitch Type)
    ]

    # Apply conditional formatting for batter table
    for row_idx, row in enumerate(final_batter_summary.values, start=1):
        for col_idx, value in enumerate(row):
            # Remove the units for comparison
            if isinstance(value, str) and ('mph' in value or 'in' in value or '%' in value):
                value = value.split()[0]

            try:
                value = float(value)
            except ValueError:
                continue

            # # Seen logic
            if col_idx == 1:  # '# Seen'
                if value >= 50:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value <= 30:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))
                    
            # Avg EV logic
            if col_idx == 2:  # 'Avg EV'
                if value >= 95:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))
                elif value <= 75:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))
    
            # Hard Hit % logic
            if col_idx == 3:  # 'Hard Hit %'
                if value >= 38:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))

            # Whiff % logic
            elif col_idx == 4:  # 'Whiff %'
                if value >= 32:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.pink))
                elif value <= 22:
                    batter_table_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.lightgreen))

    batter_table.setStyle(TableStyle(batter_table_style))
    batter_table.hAlign = 'LEFT'
    elements.append(batter_table)
    
    # Build the PDF without the notes image
    doc.build(elements)

    # Add the notes image to the PDF at a specific location
    def overlay_notes_image(temp_pdf_file, notes_image_path, output_pdf_file):
        # Create a BytesIO buffer to hold the overlay
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=landscape(letter))

        # Position the notes image at the bottom-right corner
        image_x = landscape(letter)[0] - (6 * inch)  # Adjust X-coordinate
        image_y = 0.45 * inch  # Adjust Y-coordinate
        c.drawImage(notes_image_path, image_x, image_y, width=5 * inch, height=2.5 * inch, preserveAspectRatio=True)

        # Finalize the overlay
        c.save()
        buffer.seek(0)

        # Load the overlay and base PDF
        overlay_pdf = PdfReader(buffer)
        base_pdf = PdfReader(temp_pdf_file)
        writer = PdfWriter()

        # Merge the overlay onto the first page
        base_page = base_pdf.pages[0]
        overlay_page = overlay_pdf.pages[0]
        base_page.merge_page(overlay_page)

        # Add the merged page to the writer
        writer.add_page(base_page)

        # Write the final PDF
        with open(output_pdf_file, "wb") as output_file:
            writer.write(output_file)

    # Overlay the notes image onto the final PDF
    overlay_notes_image(temp_pdf_file, notes_image_path, output_pdf_file)

    # Open the final PDF
    os.system(f'start {output_pdf_file}')

# main function for running the entire code
if __name__ == "__main__":
    try:
        # Get user input by name
        batter_name_input = input("Enter batter name (or press Enter for default): ")
        pitcher_name_input = input("Enter pitcher name (or press Enter for default): ")

        # Lookup player IDs in the MySQL database
        if batter_name_input:
            batter_id = get_player_id_by_name(batter_name_input)
            if batter_id is None:
                raise ValueError(f"Batter name '{batter_name_input}' not found.")
        else:
            batter_id = 467793  # Default batter ID (Carlos Santana)

        if pitcher_name_input:
            pitcher_id = get_player_id_by_name(pitcher_name_input)
            if pitcher_id is None:
                raise ValueError(f"Pitcher name '{pitcher_name_input}' not found.")
        else:
            pitcher_id = 666142  # Default pitcher ID (Cole Ragans)

    except ValueError as e:
        print(e)
        batter_id, pitcher_id = None, None

    # Proceed if valid batter and pitcher IDs are provided
    if batter_id is not None and pitcher_id is not None:
        try:
            # Get pitcher and batter names from the database
            pitcher_name = get_player_name(pitcher_id)
            batter_name = get_player_name(batter_id)

            # Run load_and_filter_data to get filtered data and details
            filtered_pitcher_df, filtered_batter_df, pitcher_name, batter_name, pitch_hand, batter_side = load_and_filter_data(file_path, batter_id, pitcher_id)

            # Analyze the game outcomes to generate summaries
            final_pitch_summary, final_batter_summary = analyze_game_outcomes(filtered_pitcher_df, filtered_batter_df)

            # Generate the PDF report with ReportLab
            generate_reportlab_pdf(pitcher_name, batter_name, pitch_hand, batter_side, final_pitch_summary, final_batter_summary, filtered_pitcher_df)

        except ValueError as e:
            print(f"Error during processing: {e}")
    else:
        print("Invalid input. Unable to proceed without valid pitcher and batter information.")
