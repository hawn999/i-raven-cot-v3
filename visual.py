
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

NPZ_FILE_PATH = '/home/scxhc1/nvme_data/cot_raven/I-RAVEN/in_distribute_four_out_center_single/RAVEN_8799_test.npz'
name = "view_3x5_" + Path(NPZ_FILE_PATH).stem + ".pdf"
# --------------------

# 加载数据
try:
    data = np.load(NPZ_FILE_PATH)
    img = data['image']
    target = int(data['target'])
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径 -> {NPZ_FILE_PATH}")
    exit()

# 定义网格布局
fig = plt.figure(figsize=(8, 6.9)) # 稍微加宽图形以适应5列
heights = (3, 2) # 调整上下文和答案的高度比例
outer = gridspec.GridSpec(
        2, 1,
        wspace=0.2,
        hspace=0.2,
        height_ratios=heights)

# --- 1. 绘制 3x5 上下文网格 ---
context_spec = gridspec.GridSpecFromSubplotSpec(
        3, 5, # 3行5列
        subplot_spec=outer[0],
        wspace=0.1, hspace=0.1)

# 上下文有 3*5 = 15 个格子，但只显示前14个图像
for i in range(15):
    if i < 14: # 最后一个格子是问号，不绘制
        ax = plt.Subplot(fig, context_spec[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img[i, :, :], cmap = 'gray') # 使用灰度图的反色，更清晰
        fig.add_subplot(ax)
    else: # 在最后一个格子上画一个问号
        ax = plt.Subplot(fig, context_spec[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, '?', fontsize=40, ha='center', va='center')
        fig.add_subplot(ax)

# --- 2. 绘制 2x4 答案网格 ---
answer_spec = gridspec.GridSpecFromSubplotSpec(
        2, 4, # 2行4列
        subplot_spec=outer[1],
        wspace=0.1, hspace=0.1)

# 答案有 8 个选项
for i in range(8):
    ax = plt.Subplot(fig, answer_spec[i])
    ax.set_xticks([])
    ax.set_yticks([])
    # 答案图像的索引从上下文图像之后开始（即从第14个开始）
    ax.imshow(img[14 + i, :, :], cmap = 'gray')

    # 如果是正确答案，用红色边框高亮显示
    if i == target:
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2.5)

    fig.add_subplot(ax)

# 设置标题并保存图像
fig.suptitle(f"RAVEN Problem (Correct Answer: {target+1})", fontsize=16)
fig.savefig(name)
