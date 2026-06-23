"""
Pytorch 绘图工具模块
参考 D2L 官方教程风格实现
"""

import matplotlib
matplotlib.use('TkAgg')  # 使用 Tkinter 后端
import matplotlib.pyplot as plt
import numpy as np


# === 基础设置 ===
def use_svg_display():
    """使用 SVG 格式显示图像（D2L 风格）"""
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['toolbar'] = 'None'
    # 字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['mathtext.fontset'] = 'dejavusans'


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴属性（D2L 风格）"""
    axes.set_xlabel(xlabel, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(True, linestyle='--', alpha=0.6)


class Animator:
    """动态绘制数据动画（D2L 风格）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(6, 4), title=None):  # 增大图像尺寸
        use_svg_display()
        
        plt.ion()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        
        # 设置更大的边距，确保标签不被遮挡
        self.fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.legend = legend
        self.fmts = fmts
        
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        
        self.X, self.Y = None, None
    
    def add(self, x, y):
        """向图表中添加数据点"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        if self.X is None:
            self.X = [[] for _ in range(n)]
            self.Y = [[] for _ in range(n)]
        
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        
        self.config_axes()
        
        # 添加标题
        if self.title:
            self.axes[0].set_title(self.title, fontsize=14, pad=15)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def show(self):
        """显示最终图像"""
        plt.ioff()
        plt.show()


# === 静态绘图函数 ===
def plot(X, Y, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
    """绘制简单折线图（D2L 风格）"""
    use_svg_display()
    plt.figure(figsize=figsize)
    
    if not hasattr(Y, "__len__"):
        Y = [Y]
    if not hasattr(X, "__len__"):
        X = [X] * len(Y)
    
    for x, y, fmt in zip(X, Y, fmts):
        plt.plot(x, y, fmt)
    
    set_axes(plt.gca(), xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show(block=True)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """显示图像网格（D2L 风格）"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.detach().numpy()
        except:
            pass
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show(block=True)


def plot_scatter(X, Y, xlabel=None, ylabel=None, title=None, 
                 color='blue', marker='o', figsize=(3.5, 2.5)):
    """绘制散点图（D2L 风格）"""
    use_svg_display()
    plt.figure(figsize=figsize)
    plt.scatter(X, Y, c=color, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show(block=True)


# === 便捷函数 ===
def plot_loss(train_loss, test_loss, xlabel='epoch', ylabel='loss', 
              legend=['train', 'test'], figsize=(3.5, 2.5)):
    """绘制训练损失和测试损失对比图（D2L 风格）"""
    use_svg_display()
    plt.figure(figsize=figsize)
    plt.plot(train_loss, '-', label=legend[0])
    plt.plot(test_loss, 'm--', label=legend[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show(block=True)


# 保持向后兼容的别名
d2l_plot = Animator
