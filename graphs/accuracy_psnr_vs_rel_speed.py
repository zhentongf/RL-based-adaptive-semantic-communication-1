import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义默认的 CSV 文件路径列表
DEFAULT_CSV_PATHS = [
    os.path.join(current_dir, 'transmission_cifar_data_260415.csv'),
    os.path.join(current_dir, 'transmission_cifar_data_use_nn_false_260421.csv'),
    os.path.join(current_dir, 'transmission_cifar_data_use_nn_true_260421.csv')
]

# 定义默认的输出图像路径
OUTPUT_IMAGE_PATH = os.path.join(current_dir, 'accuracy_psnr_vs_rel_speed_plot.png')

def plot_accuracy_psnr_vs_rel_speed(csv_file_paths=DEFAULT_CSV_PATHS):
    """
    读取多个 CSV 文件，按 rel_speed_values 分组计算平均值，并将对比曲线绘制在同一张图上。
    """
    # 创建绘图，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 定义不同文件的图例名称、颜色和标记
    labels = [os.path.basename(path).replace('.csv', '') for path in csv_file_paths]
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    for i, csv_path in enumerate(csv_file_paths):
        # 检查 CSV 文件是否存在
        if not os.path.exists(csv_path):
            print(f"Warning: File not found at {csv_path}, skipping...")
            continue
        
        print(f"Processing: {os.path.basename(csv_path)}...")
        
        # 加载数据
        df = pd.read_csv(csv_path)

        # 按 rel_speed_values 分组并计算平均值
        # 注意：这里假设 CSV 中存在 'rel_speed_values' 列
        grouped = df.groupby('rel_speed_values').agg({
            'accuracy': 'mean',
            'psnr': 'mean'
        }).reset_index()

        # 绘制 Accuracy vs Relative Speed 折线图
        ax1.plot(grouped['rel_speed_values'], grouped['accuracy'], 
                 marker=markers[i % len(markers)], 
                 linestyle='-', 
                 color=colors[i % len(colors)],
                 label=labels[i])
        
        # 绘制 PSNR vs Relative Speed 折线图
        ax2.plot(grouped['rel_speed_values'], grouped['psnr'], 
                 marker=markers[i % len(markers)], 
                 linestyle='-', 
                 color=colors[i % len(colors)],
                 label=labels[i])

    # 设置 Accuracy 子图属性
    ax1.set_xlabel('Relative Speed (m/s)')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy vs Relative Speed Comparison')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 设置 PSNR 子图属性
    ax2.set_xlabel('Relative Speed (m/s)')
    ax2.set_ylabel('Average PSNR (dB)')
    ax2.set_title('PSNR vs Relative Speed Comparison')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
    plt.close(fig)
    print(f"\nSuccessfully saved comparison plot to: {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    plot_accuracy_psnr_vs_rel_speed()
