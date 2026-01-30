"""
语音信号的STFT和小波变换分析
对模拟语音信号进行时频分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载信号数据
print("加载信号数据...")
data = np.load('speech_signal_data.npz')
t = data['time']
speech_signal = data['signal']
fs = int(data['sampling_rate'])

print(f"采样率: {fs} Hz")
print(f"信号长度: {len(speech_signal)} 个采样点")
print(f"持续时间: {t[-1]:.2f} 秒")

# ==================== STFT 分析 ====================
print("\n进行STFT分析...")

# STFT参数
nperseg = 256  # 窗口长度
noverlap = nperseg // 2  # 重叠长度（50%重叠）
nfft = 512  # FFT点数

# 计算STFT
f_stft, t_stft, Zxx = signal.stft(speech_signal, fs, 
                                   nperseg=nperseg, 
                                   noverlap=noverlap, 
                                   nfft=nfft,
                                   window='hann')

# 计算功率谱密度（dB）
stft_magnitude = np.abs(Zxx)
stft_db = 20 * np.log10(stft_magnitude + 1e-10)  # 添加小值避免log(0)

print(f"STFT时间分辨率: {t_stft[1] - t_stft[0]:.4f} 秒")
print(f"STFT频率分辨率: {f_stft[1] - f_stft[0]:.2f} Hz")
print(f"STFT输出形状: {Zxx.shape}")

# ==================== 小波变换分析 ====================
print("\n进行连续小波变换(CWT)分析...")

# 选择小波函数
wavelet = 'cmor1.5-1.0'  # 复Morlet小波，适合语音信号分析

# 定义分析的频率范围
freq_min = 50  # 最低频率 (Hz)
freq_max = 4000  # 最高频率 (Hz)
num_scales = 128  # 尺度数量

# 计算对应的尺度范围
# 对于cmor小波，中心频率约为1.0 Hz
central_freq = pywt.central_frequency(wavelet)
scales = central_freq * fs / np.linspace(freq_max, freq_min, num_scales)

# 执行CWT
coefficients, frequencies_cwt = pywt.cwt(speech_signal, scales, wavelet, 1/fs)

# 计算小波功率谱
cwt_power = np.abs(coefficients) ** 2

print(f"使用小波: {wavelet}")
print(f"中心频率: {central_freq:.4f}")
print(f"分析频率范围: {freq_min} - {freq_max} Hz")
print(f"CWT输出形状: {coefficients.shape}")

# ==================== 离散小波变换 (DWT) 分析 ====================
print("\n进行离散小波变换(DWT)多分辨率分析...")

# 选择小波
dwt_wavelet = 'db4'  # Daubechies 4小波
level = 6  # 分解层数

# 执行多级小波分解
coeffs = pywt.wavedec(speech_signal, dwt_wavelet, level=level)

# 获取各层系数
cA = coeffs[0]  # 近似系数
cD = coeffs[1:]  # 细节系数列表

print(f"使用小波: {dwt_wavelet}")
print(f"分解层数: {level}")
print(f"近似系数长度: {len(cA)}")
for i, cd in enumerate(cD):
    # 计算每层对应的频率范围
    freq_low = fs / (2**(level-i+1))
    freq_high = fs / (2**(level-i))
    print(f"细节系数D{level-i}长度: {len(cd)}, 频率范围: {freq_low:.1f}-{freq_high:.1f} Hz")

# ==================== 绘制结果 ====================
print("\n生成分析图像...")

# 创建主图 - 时域信号 + STFT + CWT
fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))

# 1. 时域信号
ax1 = axes1[0]
ax1.plot(t, speech_signal, linewidth=0.5, color='blue')
ax1.set_xlabel('时间 (s)', fontsize=10)
ax1.set_ylabel('幅度', fontsize=10)
ax1.set_title('原始语音信号 (时域)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, t[-1]])

# 2. STFT时频图
ax2 = axes1[1]
# 限制频率显示范围
freq_mask = f_stft <= 4000
im2 = ax2.pcolormesh(t_stft, f_stft[freq_mask], stft_db[freq_mask, :], 
                      shading='gouraud', cmap='jet')
ax2.set_xlabel('时间 (s)', fontsize=10)
ax2.set_ylabel('频率 (Hz)', fontsize=10)
ax2.set_title('短时傅里叶变换 (STFT) 时频谱图', fontsize=12)
cbar2 = plt.colorbar(im2, ax=ax2, label='功率 (dB)')
ax2.set_ylim([0, 4000])

# 3. CWT时频图
ax3 = axes1[2]
im3 = ax3.pcolormesh(t, frequencies_cwt, np.abs(coefficients), 
                      shading='gouraud', cmap='jet')
ax3.set_xlabel('时间 (s)', fontsize=10)
ax3.set_ylabel('频率 (Hz)', fontsize=10)
ax3.set_title(f'连续小波变换 (CWT) 时频谱图 (小波: {wavelet})', fontsize=12)
cbar3 = plt.colorbar(im3, ax=ax3, label='幅度')
ax3.set_ylim([freq_min, freq_max])

plt.tight_layout()
plt.savefig('stft_cwt_analysis.png', dpi=150, bbox_inches='tight')
print("STFT和CWT分析图已保存为 'stft_cwt_analysis.png'")

# 创建DWT多分辨率分析图
fig2, axes2 = plt.subplots(level + 2, 1, figsize=(14, 16))

# 原始信号
axes2[0].plot(t, speech_signal, linewidth=0.5, color='blue')
axes2[0].set_title('原始信号', fontsize=10)
axes2[0].set_xlim([0, t[-1]])
axes2[0].grid(True, alpha=0.3)

# 近似系数
t_approx = np.linspace(0, t[-1], len(cA))
axes2[1].plot(t_approx, cA, linewidth=0.5, color='green')
axes2[1].set_title(f'近似系数 A{level} (0-{fs/(2**(level+1)):.1f} Hz)', fontsize=10)
axes2[1].set_xlim([0, t[-1]])
axes2[1].grid(True, alpha=0.3)

# 细节系数
for i, cd in enumerate(cD):
    t_detail = np.linspace(0, t[-1], len(cd))
    level_num = level - i
    freq_low = fs / (2**(level_num+1))
    freq_high = fs / (2**level_num)
    axes2[i + 2].plot(t_detail, cd, linewidth=0.5, color='red')
    axes2[i + 2].set_title(f'细节系数 D{level_num} ({freq_low:.1f}-{freq_high:.1f} Hz)', fontsize=10)
    axes2[i + 2].set_xlim([0, t[-1]])
    axes2[i + 2].grid(True, alpha=0.3)

axes2[-1].set_xlabel('时间 (s)', fontsize=10)
plt.tight_layout()
plt.savefig('dwt_multiresolution.png', dpi=150, bbox_inches='tight')
print("DWT多分辨率分析图已保存为 'dwt_multiresolution.png'")

# 创建STFT和CWT对比图（2x2布局：上面时频图，下面PSD）
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))

# ===== 上排：时频谱图（xy轴调换：频率在x轴，时间在y轴）=====

# 小波时频图（左上）
ax_cwt_spec = axes3[0, 0]
# 注意：调换x和y，频率在x轴，时间在y轴
cwt_data = np.abs(coefficients).T
# 增强对比度：使用百分位数设置颜色范围
cwt_vmin = np.percentile(cwt_data, 5)
cwt_vmax = np.percentile(cwt_data, 98)
im_cwt = ax_cwt_spec.pcolormesh(frequencies_cwt, t, cwt_data, 
                                 shading='gouraud', cmap='jet',
                                 vmin=cwt_vmin, vmax=cwt_vmax)
ax_cwt_spec.set_xlabel('frequency /Hz', fontsize=10)
ax_cwt_spec.set_ylabel('time /s', fontsize=10)
ax_cwt_spec.set_title('spectrogram (wavelet)', fontsize=12)
ax_cwt_spec.set_xlim([0, 4000])
ax_cwt_spec.set_ylim([0, t[-1]])
ax_cwt_spec.invert_yaxis()  # 时间从上到下

# STFT时频图（右上）
ax_stft_spec = axes3[0, 1]
# 调换x和y
stft_data = stft_magnitude[freq_mask, :].T
# 增强对比度
stft_vmin = np.percentile(stft_data, 5)
stft_vmax = np.percentile(stft_data, 98)
im_stft = ax_stft_spec.pcolormesh(f_stft[freq_mask], t_stft, stft_data, 
                                   shading='gouraud', cmap='jet',
                                   vmin=stft_vmin, vmax=stft_vmax)
ax_stft_spec.set_xlabel('frequency /Hz', fontsize=10)
ax_stft_spec.set_ylabel('time /s', fontsize=10)
ax_stft_spec.set_title('spectrogram (STFT)', fontsize=12)
ax_stft_spec.set_xlim([0, 4000])
ax_stft_spec.set_ylim([0, t_stft[-1]])
ax_stft_spec.invert_yaxis()  # 时间从上到下

# ===== 下排：功率谱密度 (PSD) =====

# 计算小波PSD（对时间维度求平均）
cwt_psd = np.mean(np.abs(coefficients) ** 2, axis=1)

# 计算STFT PSD（对时间维度求平均）
stft_psd = np.mean(stft_magnitude ** 2, axis=1)

# 小波PSD（左下）
ax_cwt_psd = axes3[1, 0]
ax_cwt_psd.plot(frequencies_cwt, cwt_psd, linewidth=1, color='blue')
ax_cwt_psd.set_xlabel('frequency /Hz', fontsize=10)
ax_cwt_psd.set_ylabel('amplitude', fontsize=10)
ax_cwt_psd.set_title('psd (wavelet)', fontsize=12)
ax_cwt_psd.set_xlim([0, 4000])
ax_cwt_psd.grid(True, alpha=0.3)

# STFT PSD（右下）
ax_stft_psd = axes3[1, 1]
ax_stft_psd.plot(f_stft[freq_mask], stft_psd[freq_mask], linewidth=1, color='blue')
ax_stft_psd.set_xlabel('frequency /Hz', fontsize=10)
ax_stft_psd.set_ylabel('amplitude', fontsize=10)
ax_stft_psd.set_title('psd (STFT)', fontsize=12)
ax_stft_psd.set_xlim([0, 4000])
ax_stft_psd.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stft_cwt_comparison.png', dpi=150, bbox_inches='tight')
print("STFT和CWT对比图已保存为 'stft_cwt_comparison.png'")

# 创建不同小波的CWT对比
print("\n使用不同小波进行CWT分析...")
wavelets_to_compare = ['cmor1.5-1.0', 'morl', 'cgau4']
wavelet_names = ['复Morlet小波', 'Morlet小波', '复高斯小波(阶数4)']

fig4, axes4 = plt.subplots(len(wavelets_to_compare), 1, figsize=(14, 12))

for idx, (wv, wv_name) in enumerate(zip(wavelets_to_compare, wavelet_names)):
    try:
        cf = pywt.central_frequency(wv)
        sc = cf * fs / np.linspace(freq_max, freq_min, num_scales)
        coef, freq = pywt.cwt(speech_signal, sc, wv, 1/fs)
        
        ax = axes4[idx]
        im = ax.pcolormesh(t, freq, np.abs(coef), shading='gouraud', cmap='jet')
        ax.set_xlabel('时间 (s)', fontsize=10)
        ax.set_ylabel('频率 (Hz)', fontsize=10)
        ax.set_title(f'CWT - {wv_name} ({wv})', fontsize=12)
        plt.colorbar(im, ax=ax, label='幅度')
        ax.set_ylim([freq_min, freq_max])
    except Exception as e:
        print(f"小波 {wv} 处理出错: {e}")
        axes4[idx].text(0.5, 0.5, f'小波 {wv} 不可用', ha='center', va='center')

plt.tight_layout()
plt.savefig('cwt_wavelet_comparison.png', dpi=150, bbox_inches='tight')
print("不同小波CWT对比图已保存为 'cwt_wavelet_comparison.png'")

# 显示所有图像
plt.show()

# ==================== 分析总结 ====================
print("\n" + "="*60)
print("分析总结")
print("="*60)

print("\n1. STFT分析:")
print(f"   - 窗口长度: {nperseg} 采样点 ({nperseg/fs*1000:.1f} ms)")
print(f"   - 重叠: {noverlap} 采样点 ({noverlap/nperseg*100:.0f}%)")
print(f"   - FFT点数: {nfft}")
print(f"   - 时间分辨率: {(nperseg-noverlap)/fs*1000:.2f} ms")
print(f"   - 频率分辨率: {fs/nfft:.2f} Hz")

print("\n2. CWT分析 (连续小波变换):")
print(f"   - 小波函数: {wavelet}")
print(f"   - 频率范围: {freq_min} - {freq_max} Hz")
print(f"   - 尺度数量: {num_scales}")
print("   - 优势: 多分辨率分析，低频时间分辨率高，高频频率分辨率高")

print("\n3. DWT分析 (离散小波变换):")
print(f"   - 小波函数: {dwt_wavelet}")
print(f"   - 分解层数: {level}")
print("   - 各层频率范围:")
print(f"     A{level}: 0 - {fs/(2**(level+1)):.1f} Hz (低频近似)")
for i in range(level):
    level_num = level - i
    freq_low = fs / (2**(level_num+1))
    freq_high = fs / (2**level_num)
    print(f"     D{level_num}: {freq_low:.1f} - {freq_high:.1f} Hz")

print("\n4. STFT vs CWT 对比:")
print("   - STFT: 固定时间-频率分辨率（窗口长度固定）")
print("   - CWT: 自适应分辨率（低频高时间分辨率，高频高频率分辨率）")
print("   - STFT适合稳态信号，CWT适合非稳态信号如语音")

print("\n生成的图像文件:")
print("   - stft_cwt_analysis.png: 时域信号、STFT和CWT综合分析图")
print("   - dwt_multiresolution.png: DWT多分辨率分析图")
print("   - stft_cwt_comparison.png: STFT和CWT对比图")
print("   - cwt_wavelet_comparison.png: 不同小波函数CWT对比图")
