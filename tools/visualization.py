import json
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Function to load the data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to create combined bar charts for all difficulty levels in a single figure
def create_combined_bar_chart(data_dict, score_types, output_dir):
    sns.set(style="whitegrid")
    
    models = list(next(iter(data_dict.values())).keys())
    difficulties = list(data_dict.keys())
    
    fig, axes = plt.subplots(1, len(score_types), figsize=(20, 8), sharey=False)

    # Ensure 'axes' is always a list, even when there's only one score type
    if len(score_types) == 1:
        axes = [axes]
    
    for i, score_type in enumerate(score_types):
        ax = axes[i]
        score_data = {difficulty: [data_dict[difficulty][model][score_type] for model in models] for difficulty in difficulties}
        x = np.arange(len(models))
        width = 0.2
        
        # Plot bars for each difficulty level
        for j, difficulty in enumerate(difficulties):
            ax.bar(x + j * width, score_data[difficulty], width, label=f'{difficulty.capitalize()}')

        ax.set_xlabel('Models')
        ax.set_ylabel('Average ' + score_type.replace('_', ' ').capitalize() + ' Score')
        ax.set_title(f'Average {score_type.replace("_", " ").capitalize()} Score by Model and Difficulty')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'combined_scores.png')
    plt.savefig(output_path)
    plt.show()

# Paths to the JSON files
files = {
    "easy": "/Users/wad3/Downloads/Research/visual_autobench/code/document/basic_understanding/scores/easy_scores.json",
    "medium": "/Users/wad3/Downloads/Research/visual_autobench/code/document/basic_understanding/scores/medium_scores.json",
    "hard": "/Users/wad3/Downloads/Research/visual_autobench/code/document/basic_understanding/scores/hard_scores.json"
}

# Directory to save the charts
output_directory = "output_charts"
os.makedirs(output_directory, exist_ok=True)

# Load data
data_dict = {difficulty: load_data(file_path) for difficulty, file_path in files.items()}

# Create combined bar charts for subjective scores across difficulties
create_combined_bar_chart(data_dict, ["average_subjective_score"], output_directory)
