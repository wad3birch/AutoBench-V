import pandas as pd
import replicate
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import Normalize

# 读取CSV文件
file_path = '/Users/wad3/Downloads/paper/visual_autobench/code/document/spatial_understanding/hard_topic_word_degrees.csv'
df = pd.read_csv(file_path)

# 提取texts和degree
texts = df['Topic Word'].tolist()
texts = json.dumps(texts)
degrees = df['Degree'].tolist()

# 调用模型计算向量
output = replicate.run(
    "nateraw/bge-large-en-v1.5:9cf9f015a9cb9c61d1a2610659cdac4a4ca222f2d3707a68517b18c198a9add1",
    input={
        "texts": texts,
        "batch_size": 32,
        "convert_to_numpy": False,
        "normalize_embeddings": True
    }
)

# 将输出转换为数组
embeddings = np.array(output)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# 添加抖动（微小扰动）
jitter_strength = 2 # 控制抖动强度的参数
embeddings_tsne += np.random.normal(scale=jitter_strength, size=embeddings_tsne.shape)

# 创建可视化
plt.figure(figsize=(10, 8))
norm = Normalize(vmin=5, vmax=40)
# 可选的 cmap 包括:
# 连续色彩映射: 'viridis', 'inferno', 'magma', 'cividis', 'YlOrRd', 'YlGnBu', 'RdYlBu'
# 发散色彩映射: 'coolwarm', 'bwr', 'seismic', 'RdBu', 'PiYG'
# 循环色彩映射: 'hsv', 'twilight', 'twilight_shifted'
# 定性色彩映射: 'Set1', 'Set2', 'Set3', 'Paired', 'tab10', 'tab20'

scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=degrees, cmap='viridis', s=70, edgecolor='w', alpha=0.5, norm=norm)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Degree', rotation=270, labelpad=15)

plt.title('t-SNE visualization of text embeddings with jitter')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
