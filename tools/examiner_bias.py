import json
import matplotlib.pyplot as plt
import numpy as np

def load_scores(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# file_path
files = [
    'document/origin/easy_scores_origin.json',
    'document/origin/medium_scores_origin.json',
    'document/origin/hard_scores_origin.json'
]

# save data
scores = []

# load data
for file in files:
    data = load_scores(file)
    scores.append(data)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
plt.rcParams.update({'font.size': 14})
# color
colors = ['#D93F49', '#E28187', '#EBBFC2', '#D5E1E3', '#AFC9CF', '#8FB4BE', '#738CC0']

for i, score_set in enumerate(scores):
    labels = list(score_set.keys())
    x = np.arange(len(labels)) + i * 0.6
    width = 0.6

    # fetch scores
    scores_values = [score_set[model]['average-objective-score'] for model in labels]

    # print scores
    print(f"难度等级: {['Easy', 'Medium', 'Hard'][i]}, 模型得分: {scores_values}")

    bars = axs[i].bar(x, scores_values, width, color=colors[:len(labels)])

    axs[i].set_ylim(0, 1)
    axs[i].tick_params(axis='y', labelsize=12)
    axs[i].set_title(f'{["Easy", "Medium", "Hard"][i]} Difficulty',fontsize = 14)
    
    axs[i].set_xticks([])
axs[0].set_ylabel('Score',fontsize=14)

for ax in axs:
    ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.legend(bars, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=len(labels), fontsize=12)

# plt.tight_layout()  # adjust the layout
plt.subplots_adjust(top=0.743, bottom=0.258, right=0.956, left=0.046, wspace=0.1)
plt.savefig('examiner_bias.pdf', dpi=500)
# plt.show()
