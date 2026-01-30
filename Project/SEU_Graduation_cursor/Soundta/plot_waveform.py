import soundata
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 初始化数据集
dataset = soundata.initialize('urbansound8k', data_home='./UrbanSound8K/UrbanSound8K')

# 查找一个 children_playing 类别的样本
print("正在查找 children_playing 样本...")
children_playing_clip = None

for clip_id in dataset.clip_ids:
    clip = dataset.clip(clip_id)
    if clip.tags and 'children_playing' in clip.tags.labels:
        children_playing_clip = clip
        break

if children_playing_clip is None:
    print("未找到 children_playing 样本!")
else:
    print(f"找到样本: {children_playing_clip.clip_id}")
    print(f"文件路径: {children_playing_clip.audio_path}")
    
    # 加载音频数据
    y, sr = children_playing_clip.audio
    
    # 创建时间轴
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y))
    
    # 绘制时域波形
    plt.figure(figsize=(12, 4))
    plt.plot(time, y, color='steelblue', linewidth=0.5)
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.title(f'UrbanSound8K - 儿童玩耍 (children_playing)\nClip ID: {children_playing_clip.clip_id}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('children_playing_waveform.png', dpi=150)
    print(f"\n波形图已保存为: children_playing_waveform.png")
    
    # 显示图片
    plt.show()
    
    # 打印音频信息
    print(f"\n音频信息:")
    print(f"  采样率: {sr} Hz")
    print(f"  样本数: {len(y)}")
    print(f"  时长: {duration:.2f} 秒")
    print(f"  最大幅度: {np.max(np.abs(y)):.4f}")
