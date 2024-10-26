import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# plt.figure(dpi=500)  # 调整图像的dpi为300
files = ['document/basic_understanding/all_scores.csv', 
         'document/spatial_understanding/all_scores.csv', 
         'document/semantic_understanding/all_scores.csv', 
         'document/reasoning_capacity/all_scores.csv', 
         'document/atmospheric_understanding/all_scores.csv']  # 请替换为实际文件名

# 使用分隔符读取CSV文件
dataframes = [pd.read_csv(file, skiprows=1) for file in files]  # 跳过第一行

# 创建子图
# fig, axs = plt.subplots(3, 5, figsize=(15, 9))  # 3行5列

# 定义颜色列表
# colors = plt.cm.inferno(np.linspace(0, 1, 6))  # 使用viridis调色板，生成6种颜色
# # 定义颜色列表
# colors = ['#61DEFA', '#617EFA', '#61AEFA', '#61FAE4', '#7661FA', '#C3E1FF']  # 使用指定的颜色
# colors = ['#61DEFA','#617EFA', '#7661FA']
colors = ['#8fb4be','#d5e1e3','#d93f49']
# colors = ['#2878b5','#9ac9db','#c82423']

# 手动设置列标题
column_titles = ['basic', 
                 'spatial', 
                 'semantic', 
                 'reasoning', 
                 'atmospheric']  # 请根据实际情况修改

# 绘制柱状图
# for i, df in enumerate(dataframes):
#     for j, difficulty in enumerate(['easy', 'medium', 'hard']):
#         subset = df[df['Difficulty'] == difficulty]
#         axs[j, i].bar(subset['Model'], subset['Objective Score'], color=colors)  # 使用不同颜色
#         axs[j, i].set_ylim(0.5, 1)  # 设置Y轴范围从0.5开始
#         # axs[j, i].set_xticklabels(subset['Model'], rotation=45)

#          # 设置X轴标签
#         axs[j, i].set_xticks(np.arange(len(subset['Model'])))  # 设置X轴刻度
#         axs[j, i].set_xticklabels(subset['Model'], rotation=45, ha='right')  # 设置X轴标签并调整位置
        
#         # 添加网格线
#         axs[j, i].grid(axis='y', linestyle='--', alpha=0.7)  # 添加Y轴网格线

#         # 设置每列的标题
#         if j == 0:  # 只在第一行设置列标题
#             axs[j, i].set_title(column_titles[i])  # 使用手动设置的标题

#         # 设置每行的难度标题
#         if i == 0:  # 只在第一列设置行标题
#             axs[j, i].set_ylabel(difficulty)

# # 调整子图之间的距离
# plt.subplots_adjust(wspace=4, hspace=4)  # 增加水平和垂直间距

# plt.tight_layout()
# plt.show()

# 创建子图
fig, axs = plt.subplots(1, 5, figsize=(15, 5), sharey=True)  # 1行5列，共享Y轴

# 设置字体大小
plt.rcParams.update({'font.size': 14})  # 增大字体

# 绘制柱状图
for i, df in enumerate(dataframes):
    for j, difficulty in enumerate(['easy', 'medium', 'hard']):
        subset = df[df['Difficulty'] == difficulty]
        # 计算每个柱子的偏移量
        x = np.arange(len(subset['Model']))*1.2 + j * 0.4 # 每个难度的柱子偏移
        axs[i].bar(x, subset['Objective Score'], color=colors[j], width=0.4)  # 增加柱子宽度

    axs[i].set_ylim(0.5, 1)  # 设置Y轴范围从0.5开始
    axs[i].tick_params(axis='y', labelsize=14)  # 增大Y轴刻度字体大小

    # 设置每列的标题
    axs[i].set_title(column_titles[i], fontsize=14)  # 使用手动设置的标题，并增加字体大小
      # 设置Y轴标签，并增加字体大小

    # 旋转X轴标签并增加字体大小
    axs[i].set_xticks(np.arange(len(subset['Model'])) *1.2 + 0.4)  # 设置X轴刻度
    axs[i].set_xticklabels(subset['Model'], rotation=45, ha='right', fontsize=10)  # 旋转模型名称并增加字体大小

    # 添加网格线
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)  # 添加Y轴网格线
axs[0].set_ylabel('Score',fontsize=14)
handles = [plt.Rectangle((0,0),1,1, color=colors[j]) for j in range(len(['easy', 'medium', 'hard']))]
labels = ['easy', 'medium', 'hard']
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.8))  # 调整图例位置，去掉标题

# 调整子图之间的距离
plt.subplots_adjust(top=0.6, bottom=0.2, right=0.9, left=0.1, wspace=0.1)  # 增加水平间距并调整顶部间距

plt.savefig('overall.pdf', dpi=1000)  
plt.show()
# for i, df in enumerate(dataframes):
#     for j, difficulty in enumerate(['easy', 'medium', 'hard']):
#         subset = df[df['Difficulty'] == difficulty]
#         print(f"Model: {column_titles[i]}, Difficulty: {difficulty}, Objective Score: {subset['Objective Score'].values}")
