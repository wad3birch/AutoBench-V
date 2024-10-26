import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 文件路径列表
file_paths = [
    "/Users/wad3/Downloads/Research/visual_autobench/code/document/atmospheric_understanding/answers/easy_answers.json",
    "/Users/wad3/Downloads/Research/visual_autobench/code/document/atmospheric_understanding/answers/medium_answers.json",
    "/Users/wad3/Downloads/Research/visual_autobench/code/document/atmospheric_understanding/answers/hard_answers.json"
]

# 初始化数据结构存储分数
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# 读取每个json文件并提取数据
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for entry in content:
            aspect = entry["aspect"]
            level = entry["level"]
            model = entry["model"]
            subjective_score = entry["subjective_score"]
            data[model][level][aspect].append(subjective_score)

# 计算每个aspect的平均分
average_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
for model, levels in data.items():
    for level, aspects in levels.items():
        for aspect, scores in aspects.items():
            average_scores[aspect][model][level] = np.mean(scores)

# 画出柱状图，每个aspect画一张图
for aspect, models in average_scores.items():
    plt.figure(figsize=(12, 8))
    num_models = len(models)
    bar_width = 0.8 / num_models  # Adjusted bar width to fit all models in a group  # X positions for groups

    # Set distinct colors for each model
    colors = plt.cm.tab20(np.linspace(0, 1, num_models))
    
    for i, (model, levels) in enumerate(models.items()):
        levels_sorted = sorted(levels.items())
        levels_names, scores = zip(*levels_sorted)
        x = np.arange(len(levels_names))
        x_positions = x + (i - num_models // 2) * bar_width  # Adjust x positions for each model

        plt.bar(x_positions, scores, width=bar_width, label=model, color=colors[i], edgecolor='black')

    plt.xlabel('Difficulty Level', fontsize=14)
    plt.ylabel('Average Subjective Score', fontsize=14)
    plt.title(f'Average Subjective Scores for Aspect: {aspect}', fontsize=16)
    plt.xticks(x, levels_names, rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model', title_fontsize='13', fontsize='11')
    plt.tight_layout()
    plt.show()
