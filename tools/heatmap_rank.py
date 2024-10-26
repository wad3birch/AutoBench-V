import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Model': ['GLM-4v', 'Qwen2-VL', 'GPT-4o mini', 'Claude-3.5', 'GPT-4o', 'Gemini-Flash', 'Claude-3'],
    'Atmospheric Understanding Easy': [1, 2, 3, 4, 5, 6, 7],
    'Atmospheric Understanding Medium': [6, 1, 2, 7, 3, 4, 5],
    'Atmospheric Understanding Hard': [6, 2, 3, 7, 1, 4, 5],
    'Basic Understanding Easy': [1, 5, 4, 2, 3, 6, 7],
    'Basic Understanding Medium': [5, 1, 3, 4, 2, 6, 7],
    'Basic Understanding Hard': [5, 3, 4, 6, 1, 2, 7],
    'Reasoning Capacity Easy': [3, 2, 7, 5, 1, 4, 6],
    'Reasoning Capacity Medium': [4, 5, 2, 3, 1, 6, 7],
    'Reasoning Capacity Hard': [6, 3, 4, 7, 1, 5, 2],
    'Semantic Understanding Easy': [2, 3, 5, 4, 1, 6, 7],
    'Semantic Understanding Medium': [3, 2, 4, 7, 5, 6, 1],
    'Semantic Understanding Hard': [7, 3, 4, 6, 2, 5, 1],
    'Spatial Understanding Easy': [3, 5, 6, 2, 1, 4, 7],
    'Spatial Understanding Medium': [1, 2, 5, 6, 3, 4, 7],
    'Spatial Understanding Hard': [6, 3, 5, 4, 2, 1, 7]
}

df = pd.DataFrame(data)

new_model_order = [
    'GPT-4o', 'GPT-4o mini', 'Gemini-1.5-Flash', 
    'Claude-3.5-Sonnet', 'Claude-3-Haiku', 'GLM-4v', 'Qwen2-VL'
]

df['Model'] = df['Model'].map({
    'GLM-4v': 'GLM-4v',
    'Qwen2-VL': 'Qwen2-VL',
    'GPT-4o mini': 'GPT-4o mini',
    'Claude-3.5': 'Claude-3.5-Sonnet',
    'GPT-4o': 'GPT-4o',
    'Gemini-Flash': 'Gemini-1.5-Flash',
    'Claude-3': 'Claude-3-Haiku'
})

df = df.set_index('Model').loc[new_model_order].reset_index()

aspects = {
    'Basic.': ['Basic Understanding Easy', 'Basic Understanding Medium', 'Basic Understanding Hard'],
    'Spatial.': ['Spatial Understanding Easy', 'Spatial Understanding Medium', 'Spatial Understanding Hard'],
    'Reasoning.': ['Reasoning Capacity Easy', 'Reasoning Capacity Medium', 'Reasoning Capacity Hard'],
    'Semantic.': ['Semantic Understanding Easy', 'Semantic Understanding Medium', 'Semantic Understanding Hard'],
    'Atmospheric.': ['Atmospheric Understanding Easy', 'Atmospheric Understanding Medium', 'Atmospheric Understanding Hard']
}

cubehelix_cmap = sns.color_palette("crest", as_cmap=True)

fig, axs = plt.subplots(1, 5, figsize=(10, 5))

for i, (aspect, columns) in enumerate(aspects.items()):
    ax = axs[i]
    sns.heatmap(df[columns], annot=True, cmap=cubehelix_cmap, yticklabels=(df['Model'] if i == 0 else False), ax=ax, cbar=(i == 5), annot_kws={"size": 15}, square=False)
    ax.set_title(aspect, fontsize=16)  # 增大子图标题的字体大小
    ax.set_xticklabels(['E', 'M', 'H'], rotation=0, fontsize=15)
    if i == 0:
        for label in ax.get_yticklabels():
            label.set_ha('right')
            label.set_rotation(315)
            label.set_fontstyle('italic')
            label.set_fontsize(14)
    else:
        ax.set_yticks([])

plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.1)
# plt.suptitle('Heatmap of Model Rankings by Aspect', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('heatmap.pdf', dpi=500)
plt.show()
