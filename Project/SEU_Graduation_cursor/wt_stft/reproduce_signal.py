"""
复现语音信号的时域和频域图
根据提供的clean speech signal图像生成相似的信号
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 信号参数
fs = 8000  # 采样率 (Hz)
duration = 5  # 持续时间 (秒)
t = np.linspace(0, duration, int(fs * duration))

# 生成复合语音信号
# 基于语音的基频和共振峰特征
def generate_speech_like_signal(t, fs):
    """
    生成类似语音的信号
    使用多个谐波分量和调制来模拟语音特征
    """
    signal_output = np.zeros_like(t)
    
    # 基频 (约100-200 Hz，这里使用变化的基频)
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)  # 基频随时间变化
    
    # 添加多个谐波分量（模拟共振峰）
    # 共振峰1: ~500 Hz
    signal_output += 3 * np.sin(2 * np.pi * 500 * t)
    
    # 共振峰2: ~1000 Hz (主峰)
    signal_output += 5 * np.sin(2 * np.pi * 1000 * t)
    
    # 共振峰3: ~1500 Hz
    signal_output += 2.5 * np.sin(2 * np.pi * 1500 * t)
    
    # 添加一些高频分量
    signal_output += 1 * np.sin(2 * np.pi * 2000 * t)
    signal_output += 0.5 * np.sin(2 * np.pi * 2500 * t)
    
    # 添加基频和低次谐波
    signal_output += 1.5 * np.sin(2 * np.pi * f0 * t)
    signal_output += 1 * np.sin(2 * np.pi * 2 * f0 * t)
    
    # 振幅调制（模拟语音的时变特性）
    # 创建语音活动段和静音段
    envelope = np.ones_like(t)
    
    # 添加多个语音片段，增加静音间隔
    speech_segments = [
        (0.4, 0.9, 0.75),    # (开始时间, 结束时间, 相对幅度)
        (1.1, 1.7, 0.85),   
        (2.0, 2.5, 0.65),   
        (2.8, 3.3, 0.9),    # 最强段
        (3.6, 4.2, 0.8),     
        (4.5, 4.8, 0.7)     
    ]
    
    # 创建包络
    envelope = np.zeros_like(t)
    for start, end, amplitude in speech_segments:
        mask = (t >= start) & (t <= end)
        # 平滑的上升和下降
        segment_t = t[mask]
        segment_duration = end - start
        rise_time = 0.02
        fall_time = 0.02
        
        segment_envelope = np.ones_like(segment_t) * amplitude
        
        # 上升沿
        rise_mask = segment_t < (start + rise_time)
        segment_envelope[rise_mask] = ((segment_t[rise_mask] - start) / rise_time) * amplitude
        
        # 下降沿
        fall_mask = segment_t > (end - fall_time)
        segment_envelope[fall_mask] = ((end - segment_t[fall_mask]) / fall_time) * amplitude
        
        # 在片段内部添加随机幅度变化（模拟音节的能量起伏）
        internal_variation = 0.2 * np.random.randn(len(segment_t))
        internal_variation = np.convolve(internal_variation, np.ones(50)/50, mode='same')  # 平滑处理
        segment_envelope = segment_envelope * (1 + internal_variation * 0.3)
        segment_envelope = np.clip(segment_envelope, 0, amplitude)
        
        envelope[mask] = segment_envelope
    
    # 添加快速振幅变化（模拟音节的韵律变化）
    syllable_modulation = 0.4 * np.sin(2 * np.pi * 4 * t) * np.sin(2 * np.pi * 0.8 * t) + 0.7
    envelope = envelope * syllable_modulation
    
    # 应用包络
    signal_output = signal_output * envelope
    
    # 添加少量噪声
    noise = np.random.normal(0, 0.3, len(t))
    signal_output += noise
    
    return signal_output

# 生成信号
speech_signal = generate_speech_like_signal(t, fs)

# 计算频谱
frequencies = np.fft.rfftfreq(len(speech_signal), 1/fs)
fft_values = np.fft.rfft(speech_signal)
magnitude_spectrum = np.abs(fft_values) / len(speech_signal)

# 绘制结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 时域图
ax1.plot(t, speech_signal, linewidth=0.5)
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.set_title('Clean speech signal', fontsize=12, color='blue')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 5])
ax1.set_ylim([-20, 20])

# 频域图
ax2.plot(frequencies, magnitude_spectrum, linewidth=0.8, color='red')
ax2.set_xlabel('Frequency (Hz)', fontsize=10)
ax2.set_ylabel('Magnitude', fontsize=10)
ax2.set_title('Clean speech signal', fontsize=12, color='blue')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 4000])

plt.tight_layout()
plt.savefig('reproduced_signal.png', dpi=150, bbox_inches='tight')
print("图像已保存为 'reproduced_signal.png'")
plt.show()

# 保存信号数据
np.savez('speech_signal_data.npz', 
         time=t, 
         signal=speech_signal, 
         frequencies=frequencies, 
         magnitude_spectrum=magnitude_spectrum,
         sampling_rate=fs)
print("信号数据已保存为 'speech_signal_data.npz'")

# 打印信号统计信息
print("\n信号统计信息:")
print(f"采样率: {fs} Hz")
print(f"持续时间: {duration} 秒")
print(f"信号长度: {len(speech_signal)} 个采样点")
print(f"幅度范围: [{speech_signal.min():.2f}, {speech_signal.max():.2f}]")
print(f"均值: {speech_signal.mean():.4f}")
print(f"标准差: {speech_signal.std():.4f}")
print(f"\n主要频率分量峰值:")
# 找出前5个最大的频率分量
peak_indices = np.argsort(magnitude_spectrum)[-10:][::-1]
for idx in peak_indices[:5]:
    if frequencies[idx] < 4000:
        print(f"  {frequencies[idx]:.1f} Hz: {magnitude_spectrum[idx]:.4f}")
