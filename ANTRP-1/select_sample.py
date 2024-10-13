import json
with open('/mnt/sda/feilongtang/Hallucination/SID/results/log/llava-1.5/beam/ours_outputs.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 获取包含hallucination_idxs的image_id，并按hallucination_idxs的长度排序
result = sorted(
    [item['image_id'] for item in data["sentences"] if item['hallucination_idxs']],
    key=lambda id: len(next(item['hallucination_idxs'] for item in data["sentences"] if item['image_id'] == id))
)
print(result[:100])