%% 清理环境
clc; clear; close all;

%% 1. 信号生成参数设置
fs = 3000;              % 采样率 (代码设置为3000Hz)
T = 1;                  % 信号时长 1秒
t = 0:1/fs:T-1/fs;      % 时间向量

%% 2. 构建两个典型的测试信号

% --- 信号 A: 线性调频信号 (Chirp) - 频率随时间增长 ---
% [修改点] 将信号幅度降低为 0.5
sig_a1 = 0.5 * sin(2*pi * (100*t + ((200-100)/(2*T)) * t.^2));
sig_a2 = 0.5 * sin(2*pi * (300*t + ((550-300)/(2*T)) * t.^2));
sig_a3 = 0.5 * sin(2*pi * (600*t + ((900-600)/(2*T)) * t.^2));

% [修改点] 添加背景噪声
noise_bg_a = 0.5 * (2 * rand(size(t)) - 1); 
sig_varying = sig_a1 + sig_a2 + sig_a3 + noise_bg_a;

% --- 信号 B: 白噪声背景下的瞬态突变 (小波变换 WT 的强项) ---
noise_bg = 2 * rand(size(t)) - 1; 

% 添加突变脉冲
pulse_center1 = 0.3;
pulse_center2 = 0.7;
width = 0.002; 

% 合成信号 B
sig_transient = noise_bg + ...
    3 * exp(-(t-pulse_center1).^2 / (2*width^2)) .* sin(2*pi*800*t) + ...
    3 * exp(-(t-pulse_center2).^2 / (2*width^2)) .* sin(2*pi*800*t);

%% 3. 进行分析与绘图 (3行2列布局)

figure('Name', '时域 vs STFT vs 小波变换 (自定义淡绿-红配色)', 'Color', 'w', 'Position', [50, 50, 1200, 900]);

% [新增] 自定义颜色映射 (Colormap): 淡绿 -> 黄 -> 红
% 这符合您描述的"幅度低偏淡绿色，幅度高偏红色"
n_colors = 256;
half_n = floor(n_colors/2);

% 前半段: 淡绿 ([0.7, 1, 0.7]) 过渡到 黄 ([1, 1, 0])
% R: 0.7 -> 1
% G: 1.0 -> 1
% B: 0.7 -> 0
r_low = linspace(0.7, 1, half_n);
g_low = linspace(1, 1, half_n);
b_low = linspace(0.7, 0, half_n);

% 后半段: 黄 ([1, 1, 0]) 过渡到 红 ([1, 0, 0])
% R: 1 -> 1
% G: 1 -> 0
% B: 0 -> 0
r_high = linspace(1, 1, n_colors - half_n);
g_high = linspace(1, 0, n_colors - half_n);
b_high = linspace(0, 0, n_colors - half_n);

% 组合成完整的 Colormap
custom_cmap = [ [r_low, r_high]', [g_low, g_high]', [b_low, b_high]' ];

% 应用自定义颜色到当前 Figure
colormap(custom_cmap);


% --- 公共 STFT 参数设置 (显式加窗) ---
win_len = 128;          
window = hamming(win_len); 
noverlap = 120;
nfft = 1024;

% ==========================================================
% 左侧列：信号 A (频率随时间增长)
% ==========================================================

% --- 1. 时域图 ---
subplot(3,2,1);
plot(t, sig_varying, 'b');
title('1. 信号 A (时域): 含噪的扫频波形');
xlabel('时间 (s)'); ylabel('幅值'); grid on;
xlim([0, 0.5]); 

% --- 2. STFT ---
subplot(3,2,3);
[s, f, t_stft] = spectrogram(sig_varying, window, noverlap, nfft, fs);
imagesc(t_stft, f, abs(s)); 
axis xy; 
ylim([0 1000]);
title('3. 信号 A (STFT): 背景为淡绿色(低能量)，线条为红黄色(高能量)');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar; 

% --- 3. 小波变换 ---
subplot(3,2,5);
[wt, f_cwt] = cwt(sig_varying, fs); 
surface(t, f_cwt, abs(wt));
shading interp; axis tight; ylim([0 1000]);
title('5. 信号 A (小波): 高频轨迹依然较粗');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar; 


% ==========================================================
% 右侧列：信号 B (白噪声 + 突变)
% ==========================================================

% --- 1. 时域图 ---
subplot(3,2,2);
plot(t, sig_transient); 
hold on;
plot(t(t>0.29 & t<0.31), sig_transient(t>0.29 & t<0.31), 'r'); 
plot(t(t>0.69 & t<0.71), sig_transient(t>0.69 & t<0.71), 'r');
hold off;
title('2. 信号 B (时域): 白噪声掩盖下的突变');
xlabel('时间 (s)'); ylabel('幅值'); grid on;
ylim([-4 4]); 

% --- 2. STFT ---
subplot(3,2,4);
[s2, f2, t_stft2] = spectrogram(sig_transient, window, noverlap, nfft, fs);
imagesc(t_stft2, f2, abs(s2)); 
axis xy; ylim([0 1000]);
title('4. 信号 B (STFT): 突变点在噪声中模糊不清');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar; 

% --- 3. 小波变换 ---
subplot(3,2,6);
[wt2, f_cwt2] = cwt(sig_transient, fs);
surface(t, f_cwt2, abs(wt2));
shading interp; axis tight; ylim([0 1000]);
title('6. 信号 B (小波): 即使有噪声，突变点依然锐利');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar; 

%% 结果说明
fprintf('绘图完成。\n');
fprintf('颜色已调整为：低幅度(背景)显示为淡绿色，高幅度(信号)显示为红色。\n');
fprintf('这种配色方案（淡绿->黄->红）通常用于强调风险或强度等级，对比度很高。\n');