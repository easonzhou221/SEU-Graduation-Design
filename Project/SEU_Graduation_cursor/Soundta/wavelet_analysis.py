import soundata
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal

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
    exit()

print(f"找到样本: {children_playing_clip.clip_id}")
print(f"文件路径: {children_playing_clip.audio_path}")

# 加载音频数据
y, sr = children_playing_clip.audio
duration = len(y) / sr
time = np.linspace(0, duration, len(y))

print(f"采样率: {sr} Hz")
print(f"音频时长: {duration:.2f} 秒")
print(f"样本数: {len(y)}")

# ============================================================
# 小波变换时频分析 (Continuous Wavelet Transform - CWT)
# ============================================================
print("\n正在进行小波变换...")

# 为了加快计算速度，对音频进行下采样
downsample_factor = 4
y_downsampled = signal.decimate(y, downsample_factor)
sr_downsampled = sr // downsample_factor
time_downsampled = np.linspace(0, duration, len(y_downsampled))

# 设置小波参数
wavelet = 'cmor1.5-1.0'  # 复Morlet小波
# 定义要分析的频率范围 (20Hz - 8000Hz)
frequencies = np.logspace(np.log10(20), np.log10(sr_downsampled/2), 128)
scales = pywt.frequency2scale(wavelet, frequencies / sr_downsampled)

# 计算连续小波变换
coefficients, freqs = pywt.cwt(y_downsampled, scales, wavelet, sampling_period=1/sr_downsampled)

# 计算小波功率谱 (scalogram)
power = np.abs(coefficients) ** 2

print("小波变换完成!")

# ============================================================
# 功率谱密度 (PSD) 计算
# ============================================================
print("正在计算功率谱密度 (PSD)...")

# 使用Welch方法计算PSD
nperseg = min(4096, len(y))
f_psd, psd = signal.welch(y, sr, nperseg=nperseg)

print("小波 PSD 计算完成!")

# ============================================================
# STFT 短时傅里叶变换分析
# ============================================================
print("\n正在进行STFT分析...")

# STFT 参数
nperseg_stft = 1024  # 窗口长度
noverlap_stft = nperseg_stft // 2  # 50% 重叠
nfft_stft = 2048  # FFT点数

# 计算 STFT
f_stft, t_stft, Zxx = signal.stft(y, sr, nperseg=nperseg_stft, 
                                   noverlap=noverlap_stft, nfft=nfft_stft)

# 计算 STFT 功率谱 (dB)
stft_power = np.abs(Zxx) ** 2
stft_power_db = 10 * np.log10(stft_power + 1e-10)

print("STFT分析完成!")

# 计算基于 STFT 的 PSD（对时间轴取平均）
stft_psd = np.mean(stft_power, axis=1)

# ============================================================
# 归一化处理 - 使两种方法的功率可比
# ============================================================
print("\n正在进行归一化处理...")

# 方法：归一化到各自的最大值 (0 dB = 最大功率)
# 这样可以比较相对功率分布，而不是绝对数值

# STFT 归一化
stft_power_normalized = stft_power / np.max(stft_power)
stft_power_db_norm = 10 * np.log10(stft_power_normalized + 1e-10)

# 小波功率归一化
power_normalized = power / np.max(power)
power_db_norm = 10 * np.log10(power_normalized + 1e-10)

print(f"STFT 原始功率范围: {10*np.log10(stft_power.min()+1e-10):.1f} ~ {10*np.log10(stft_power.max()):.1f} dB")
print(f"小波 原始功率范围: {10*np.log10(power.min()+1e-10):.1f} ~ {10*np.log10(power.max()):.1f} dB")
print("归一化后两者范围统一为: -60 ~ 0 dB (相对于各自最大值)")

# ============================================================
# 绘图 - STFT vs 小波变换 对比图
# ============================================================
print("\n正在生成对比图表...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 设置统一的颜色范围 (归一化后)
vmin, vmax = -60, 0

# ============================================================
# 第一行：时频图对比
# ============================================================

# 左上: STFT 时频图
ax1 = axes[0, 0]
im1 = ax1.pcolormesh(t_stft, f_stft, stft_power_db_norm, 
                      shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
ax1.set_ylabel('频率 (Hz)')
ax1.set_xlabel('时间 (秒)')
ax1.set_title('STFT 时频图 (Spectrogram)')
ax1.set_ylim([20, sr/2])
ax1.set_yscale('log')
ax1.set_xlim([0, duration])
cbar1 = plt.colorbar(im1, ax=ax1, label='相对功率 (dB)')

# 右上: 小波时频图 (Scalogram)
ax2 = axes[0, 1]
im2 = ax2.pcolormesh(time_downsampled, frequencies, power_db_norm, 
                      shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
ax2.set_ylabel('频率 (Hz)')
ax2.set_xlabel('时间 (秒)')
ax2.set_title('小波时频图 (Scalogram) - 复Morlet小波')
ax2.set_ylim([20, sr_downsampled/2])
ax2.set_yscale('log')
ax2.set_xlim([0, duration])
cbar2 = plt.colorbar(im2, ax=ax2, label='相对功率 (dB)')

# ============================================================
# 第二行：PSD 对比
# ============================================================

# 左下: STFT PSD
ax3 = axes[1, 0]
ax3.semilogy(f_stft, stft_psd, color='steelblue', linewidth=1.2, label='STFT PSD')
ax3.set_xlabel('频率 (Hz)')
ax3.set_ylabel('功率谱密度')
ax3.set_title('功率谱密度 (PSD) - STFT方法')
ax3.grid(True, alpha=0.3, which='both')
ax3.set_xlim([0, sr/2])
ax3.legend()

# 右下: Welch PSD (基于小波分析时已计算)
ax4 = axes[1, 1]
ax4.semilogy(f_psd, psd, color='darkgreen', linewidth=1.2, label='Welch PSD')
ax4.set_xlabel('频率 (Hz)')
ax4.set_ylabel('功率谱密度')
ax4.set_title('功率谱密度 (PSD) - Welch方法')
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xlim([0, sr/2])
ax4.legend()

# 添加总标题
fig.suptitle(f'STFT vs 小波变换 时频分析对比\n儿童玩耍 (children_playing) - Clip ID: {children_playing_clip.clip_id}', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# 保存图片
output_filename = 'stft_vs_wavelet_comparison.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存为: {output_filename}")

# 显示图片
plt.show()

# 打印分析摘要
print("\n" + "=" * 60)
print("STFT vs 小波变换 分析对比摘要")
print("=" * 60)
print(f"音频文件: {children_playing_clip.clip_id}")
print(f"采样率: {sr} Hz")
print(f"时长: {duration:.2f} 秒")
print()
print("STFT 参数:")
print(f"  窗口长度: {nperseg_stft} 样本 ({nperseg_stft/sr*1000:.1f} ms)")
print(f"  FFT点数: {nfft_stft}")
print(f"  重叠: {noverlap_stft} 样本 (50%)")
print(f"  频率分辨率: {sr/nfft_stft:.1f} Hz")
print(f"  时间分辨率: {(nperseg_stft-noverlap_stft)/sr*1000:.1f} ms")
print()
print("小波变换参数:")
print(f"  小波类型: 复Morlet小波 (cmor1.5-1.0)")
print(f"  频率范围: 20 Hz - {sr_downsampled//2} Hz")
print(f"  尺度数: {len(scales)}")
print()
print("对比说明:")
print("  - STFT: 固定时频分辨率，适合分析稳态信号")
print("  - 小波: 多分辨率分析，低频时间分辨率低但频率分辨率高，高频相反")
