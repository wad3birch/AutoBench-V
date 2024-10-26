import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
def load_scores(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 文件路径
files = [
    'document/origin/easy_scores_origin.json',
    'document/origin/medium_scores_origin.json',
    'document/origin/hard_scores_origin.json'
]

# 存储数据
scores = []

# 加载数据
for file in files:
    data = load_scores(file)
    scores.append(data)

# 创建子图，设置共享 y 轴
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
plt.rcParams.update({'font.size': 14})
# 定义颜色
colors = ['#D93F49', '#E28187', '#EBBFC2', '#D5E1E3', '#AFC9CF', '#8FB4BE', '#738CC0']

# 绘制每个难度的得分柱状图
for i, score_set in enumerate(scores):
    labels = list(score_set.keys())
    x = np.arange(len(labels)) + i * 0.6 # 模型的索引
    width = 0.6  # 增大柱子的宽度以让它们更紧密

    # 提取每个模型的得分
    scores_values = [score_set[model]['average-objective-score'] for model in labels]

    # 打印具体的数据
    print(f"难度等级: {['Easy', 'Medium', 'Hard'][i]}, 模型得分: {scores_values}")

    # 绘制得分
    bars = axs[i].bar(x, scores_values, width, color=colors[:len(labels)])  # 不偏移，柱子紧靠在x轴刻度上

    axs[i].set_ylim(0, 1)  # 设置 y 轴范围
    axs[i].tick_params(axis='y', labelsize=12)
    axs[i].set_title(f'{["Easy", "Medium", "Hard"][i]} Difficulty',fontsize = 14)
    
    # 隐藏 x 轴标签
    axs[i].set_xticks([])
axs[0].set_ylabel('Score',fontsize=14)
# 添加网格
for ax in axs:
    ax.grid(axis='y', linestyle='--', alpha=0.7)
# 添加共享图例
fig.legend(bars, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=len(labels), fontsize=12)

# plt.tight_layout()  # 自动调整子图间距
plt.subplots_adjust(top=0.743, bottom=0.258, right=0.956, left=0.046, wspace=0.1)
plt.savefig('examiner_bias.pdf', dpi=500)
# plt.show()
