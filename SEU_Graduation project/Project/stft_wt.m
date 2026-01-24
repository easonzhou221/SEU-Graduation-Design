%% 清理环境
clc; clear; close all;

%% 1. 信号生成参数设置
fs = 2000;              % 采样率 2000Hz
T = 1;                  % 信号时长 1秒
t = 0:1/fs:T-1/fs;      % 时间向量

%% 2. 构建两个典型的测试信号

% --- 信号 A: 稳态谐波信号 (STFT 的强项) ---
% 由 100Hz, 300Hz, 600Hz 三个持续的正弦波叠加而成
sig_steady = sin(2*pi*100*t) + sin(2*pi*300*t) + sin(2*pi*600*t);

% --- 信号 B: 瞬态突变信号 (小波变换 WT 的强项) ---
% 背景是低频 30Hz，但在 t=0.3s 和 t=0.7s 处有两个极短的高频脉冲(突变)
sig_transient = sin(2*pi*30*t); 
% 添加突变脉冲 (高斯调制的 800Hz 脉冲)
pulse_center1 = 0.3;
pulse_center2 = 0.7;
width = 0.002; % 脉冲非常窄
sig_transient = sig_transient + ...
    1.5 * exp(-(t-pulse_center1).^2 / (2*width^2)) .* sin(2*pi*800*t) + ...
    1.5 * exp(-(t-pulse_center2).^2 / (2*width^2)) .* sin(2*pi*800*t);

%% 3. 进行分析与绘图

figure('Name', 'STFT vs 小波变换对比', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

% ==========================================
% 第一行：稳态信号分析 (Signal A)
% ==========================================

% --- 1.1 STFT 分析 (稳态) ---
subplot(2,2,1);
% 使用汉明窗，窗长 128 (约64ms)，重叠 120
window = 128;
noverlap = 120;
nfft = 1024;
[s, f, t_stft] = spectrogram(sig_steady, window, noverlap, nfft, fs);
imagesc(t_stft, f, abs(s)); 
axis xy; colormap jet;
ylim([0 800]); % 只看 0-800Hz
title({'信号 A (稳态) - STFT 分析', '清晰的水平线条 (频率分辨率好)'});
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;

% --- 1.2 小波变换 (CWT) 分析 (稳态) ---
subplot(2,2,2);
% 使用 'amor' (Morlet) 小波
[wt, f_cwt] = cwt(sig_steady, fs); 
% 绘图 (取绝对值看能量)
surface(t, f_cwt, abs(wt));
shading interp; axis tight;
ylim([0 800]);
title({'信号 A (稳态) - 小波变换', '线条较粗/模糊 (高频处频率分辨率差)'});
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;


% ==========================================
% 第二行：瞬态突变信号分析 (Signal B)
% ==========================================

% --- 2.1 STFT 分析 (瞬态) ---
subplot(2,2,3);
[s2, f2, t_stft2] = spectrogram(sig_transient, window, noverlap, nfft, fs);
imagesc(t_stft2, f2, abs(s2)); 
axis xy;
ylim([0 1000]);
title({'信号 B (瞬态) - STFT 分析', '时间模糊 (由于窗长固定，突变被"抹宽"了)'});
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;

% --- 2.2 小波变换 (CWT) 分析 (瞬态) ---
subplot(2,2,4);
[wt2, f_cwt2] = cwt(sig_transient, fs);
surface(t, f_cwt2, abs(wt2));
shading interp; axis tight;
ylim([0 1000]);
title({'信号 B (瞬态) - 小波变换', '极高的时间精度 (像针一样定位突变)'});
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;

%% 结果说明打印
fprintf('绘图完成。\n请观察 Figure 窗口：\n');
fprintf('1. 第一行对比：STFT 的水平线条应该比小波更细、更清晰。\n');
fprintf('2. 第二行对比：小波变换能把两个突变点画得像“针”一样细，而 STFT 会把它们画成宽宽的“柱子”。\n');