import json
import replicate

# 读取JSON文件并提取aspect字段
with open('/Users/wad3/Downloads/Research/visual_autobench/code/document/basic_understanding/basic_understanding_aspects.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有aspect字段
aspects = [item['aspect'] for item in data]
texts = ",".join(aspects)
print(texts)

# 使用提取的aspect字段作为文本输入
# output = replicate.run(
#     "nateraw/bge-large-en-v1.5:9cf9f015a9cb9c61d1a2610659cdac4a4ca222f2d3707a68517b18c198a9add1",
#     input={
#         "texts": texts,
#         "batch_size": 32,
#         "convert_to_numpy": False,
#         "normalize_embeddings": True
#     }
# )

# print(output)

output = replicate.run(
    "nateraw/bge-large-en-v1.5:9cf9f015a9cb9c61d1a2610659cdac4a4ca222f2d3707a68517b18c198a9add1",
    input={
        "texts": "[\"In the water, fish are swimming.\", \"Fish swim in the water.\", \"A book lies open on the table.\"]",
        "batch_size": 32,
        "convert_to_numpy": False,
        "normalize_embeddings": True
    }
)
print(output)