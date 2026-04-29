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

# 定义默认的输出图像路径列表
DEFAULT_IMAGE_PATHS = [
    os.path.join(current_dir, 'accuracy_psnr_vs_distance_260415.png'),
    os.path.join(current_dir, 'accuracy_psnr_vs_distance_use_nn_false_260421.png'),
    os.path.join(current_dir, 'accuracy_psnr_vs_distance_use_nn_true_260421.png')
]

def plot_accuracy_psnr_vs_distance(csv_file_paths=DEFAULT_CSV_PATHS, output_image_paths=DEFAULT_IMAGE_PATHS):
    """
    循环读取多个 CSV 文件并生成对应的 accuracy 与 psnr 随距离变化的散点图。
    """
    for csv_path, img_path in zip(csv_file_paths, output_image_paths):
        # 检查 CSV 文件是否存在
        if not os.path.exists(csv_path):
            print(f"Warning: File not found at {csv_path}, skipping...")
            continue
        
        print(f"Processing: {os.path.basename(csv_path)}...")
        
        # 加载数据
        df = pd.read_csv(csv_path)

        # 创建绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy vs distance_values 散点图
        ax1.scatter(df['distance_values'], df['accuracy'], alpha=0.6, s=15, c='blue', edgecolors='none')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Accuracy\n({os.path.basename(csv_path)})')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # PSNR vs distance_values 散点图
        ax2.scatter(df['distance_values'], df['psnr'], alpha=0.6, s=15, c='orange', edgecolors='none')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title(f'PSNR\n({os.path.basename(csv_path)})')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close(fig)  # 关闭图形释放内存
        print(f"Successfully saved plot to: {img_path}")

if __name__ == '__main__':
    plot_accuracy_psnr_vs_distance()
