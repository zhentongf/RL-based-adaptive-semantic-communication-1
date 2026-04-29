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
OUTPUT_IMAGE_PATH = os.path.join(current_dir, 'snr_accuracy_psnr_plot.png')

def plot_transmission_cifar_data_snr_accuracy_psnr(csv_file_paths=DEFAULT_CSV_PATHS):
    """
    读取多个 CSV 文件，将三份数据的 Accuracy 和 PSNR 随 Model SNR 变化的曲线绘制在同一张图上进行对比。
    """
    # 创建绘图，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 定义不同文件的图例名称（取文件名部分）
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

        # 按 model_snr 分组并计算平均值
        grouped = df.groupby('model_snr').agg({
            'accuracy': 'mean',
            'psnr': 'mean'
        }).reset_index()

        # 绘制 Accuracy 曲线
        ax1.plot(grouped['model_snr'], grouped['accuracy'], 
                 marker=markers[i % len(markers)], 
                 linestyle='-', 
                 color=colors[i % len(colors)],
                 label=labels[i])
        
        # 绘制 PSNR 曲线
        ax2.plot(grouped['model_snr'], grouped['psnr'], 
                 marker=markers[i % len(markers)], 
                 linestyle='-', 
                 color=colors[i % len(colors)],
                 label=labels[i])

    # 设置 Accuracy 子图属性
    ax1.set_xlabel('Model SNR')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Accuracy')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 设置 PSNR 子图属性
    ax2.set_xlabel('Model SNR')
    ax2.set_ylabel('Average PSNR (dB)')
    ax2.set_title('PSNR')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
    plt.close(fig)
    print(f"\nSuccessfully saved comparison plot to: {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    plot_transmission_cifar_data_snr_accuracy_psnr()
