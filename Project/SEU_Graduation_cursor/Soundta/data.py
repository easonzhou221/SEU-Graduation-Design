import soundata

# 初始化 UrbanSound8K 数据集（使用本地已下载的数据）
dataset = soundata.initialize('urbansound8k', data_home='./UrbanSound8K/UrbanSound8K')

# 查看数据集基本信息
print("=" * 50)
print("数据集信息")
print("=" * 50)
print(f"数据集名称: {dataset.name}")
print(f"数据路径: {dataset.data_home}")

# 索引文件已下载完成

# 获取所有 clip IDs
clip_ids = dataset.clip_ids
print(f"总样本数: {len(clip_ids)}")
print(f"前10个 clip IDs: {clip_ids[:10]}")
print()

# 查看一个随机样本的详细信息
print("=" * 50)
print("随机样本示例")
print("=" * 50)
example_clip = dataset.choice_clip()
print(f"Clip ID: {example_clip.clip_id}")
print(f"音频文件路径: {example_clip.audio_path}")
print(f"标签 (Tags): {example_clip.tags}")
print(f"Fold: {example_clip.fold}")
print()

# 加载音频数据
print("=" * 50)
print("音频数据")
print("=" * 50)
y, sr = example_clip.audio
print(f"采样率 (Sample Rate): {sr} Hz")
print(f"音频长度: {len(y)} 样本")
print(f"音频时长: {len(y)/sr:.2f} 秒")
print()

# 查看数据集中的所有类别
print("=" * 50)
print("数据集类别统计")
print("=" * 50)
from collections import Counter
all_tags = []
for clip_id in clip_ids[:100]:  # 先看前100个样本
    clip = dataset.clip(clip_id)
    if clip.tags:
        all_tags.extend(clip.tags.labels)

tag_counts = Counter(all_tags)
print("前100个样本的类别分布:")
for tag, count in tag_counts.most_common():
    print(f"  {tag}: {count}")
