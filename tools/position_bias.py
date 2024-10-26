import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
def load_scores(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 文件路径
files = [
    ['document/scores/easy_scores_A.json', 'document/scores/easy_scores.json', 'document/scores/easy_scores_D.json'],
    ['document/scores/medium_scores_A.json', 'document/scores/medium_scores.json', 'document/scores/medium_scores_D.json'],
    ['document/scores/hard_scores_A.json', 'document/scores/hard_scores.json', 'document/scores/hard_scores_D.json']
]

# 存储数据
scores = []

# 加载数据
for file_set in files:
    score_set = { 'A': {}, 'Uniform distribution': {}, 'D': {} }
    for i, file in enumerate(file_set):
        data = load_scores(file)
        for key in data:
            score_set[list(score_set.keys())[i]][key] = data[key]['average-objective-score']
    scores.append(score_set)

# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
plt.rcParams.update({'font.size': 14})

# 绘制每个难度的偏差率柱状图
for i, score_set in enumerate(scores):
    labels = list(score_set['A'].keys())
    x = np.arange(len(labels))  # 模型的索引
    width = 0.35  # 柱子的宽度

    # 计算偏差率
    deviation_A = [(score_set['A'][model] - score_set['Uniform distribution'][model]) / score_set['Uniform distribution'][model] for model in labels]
    deviation_D = [(score_set['D'][model] - score_set['Uniform distribution'][model]) / score_set['Uniform distribution'][model] for model in labels]

    # 绘制偏差率
    axs[i].bar(x - width/2, deviation_A, width, label='"A"-Deviation rate', color='#D93F49')
    axs[i].bar(x + width/2, deviation_D, width, label='"D"-Deviation rate', color='#93CAFF')

    axs[i].set_xticks(x)
    axs[i].set_xticklabels(labels, rotation=45, fontsize=14)
    axs[i].set_ylim(-0.3, 0.3)  # 偏差率范围
    axs[i].tick_params(axis='y', labelsize=12) 
    # axs[i].set_title(f'Difficulty Level {i + 1}')
    # print(f"偏差率（A）：{deviation_A}")
    # print(f"偏差率（D）：{deviation_D}")

    # 添加网格
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)

    # axs[i].set_ylabel('Deviation rate',fontsize=14)
    if i == 0:
        axs[i].set_title('Easy Difficulty')
    elif i == 1:
        axs[i].set_title('Medium Difficulty')
    elif i ==2:
        axs[i].set_title('Hard Difficulty')
    # if i > 0:
    #     axs[i].yaxis.set_visible(False)
    axs[i].axhline(0, color='black', linewidth=0.8, linestyle='--')  # 添加水平线表示0

axs[0].set_ylabel('Deviation rate',fontsize=14)
 # 调整子图的left，right，top，bottom
handles, labels = axs[0].get_legend_handles_labels()  # 从第一个子图获取图例
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2)
plt.subplots_adjust(top=0.7, bottom=0.258, right=0.965, left=0.085, wspace=0.1)
plt.savefig('deviation.pdf',dpi = 500)
plt.show()