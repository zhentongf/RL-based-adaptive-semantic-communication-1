import csv
import random

def generate_car_sim_data(number_cars=20, road_length=5000, road_width=10, right_cars=10, left_cars=10):
    """
    生成车辆行驶模拟数据并保存为 CSV 文件。
    
    参数:
    number_cars (int): 车辆总数 (>= 2)
    road_length (int): 道路长度 (>= 100 且为 100 的倍数)
    road_width (int): 道路宽度 (>= 10)
    right_cars (int): 向右行驶的车数量 (>= 1)
    left_cars (int): 向左行驶的车数量 (>= 1)
    """
    
    # 第一步：检查输入参数
    if number_cars < 2:
        print("错误：number_cars 必须大于等于 2")
        return
    if road_length < 100 or road_length % 100 != 0:
        print("错误：road_length 必须大于等于 100 且为 100 的倍数")
        return
    if road_width < 10:
        print("错误：road_width 必须大于等于 10")
        return
    if right_cars < 1 or left_cars < 1:
        print("错误：right_cars 和 left_cars 都必须大于等于 1")
        return
    if right_cars + left_cars != number_cars:
        print(f"错误：right_cars ({right_cars}) 和 left_cars ({left_cars}) 之和必须等于 number_cars ({number_cars})")
        return

    # 第二步：计算时间点数
    # 最大速度为 50 m/s，所需时间点数为 road_length / 50 + 1 (包括起始时间 0)
    max_speed = 50
    duration = int(road_length / max_speed)
    num_time_points = duration + 1

    # 第三步：生成车辆初始属性
    cars = []
    speeds_pool = [10, 20, 30, 40, 50]
    snr_pool = [20, 30, 40]
    random.seed(42)

    # 生成向右行驶的车 (direction = 0, y = 0, x_start = 0)
    for i in range(1, right_cars + 1):
        car_id = f"v{len(cars) + 1}"
        speed = random.choice(speeds_pool)
        snr = random.choice(snr_pool)
        cars.append({
            "car_ID": car_id,
            "speed": speed,
            "direction": 0,
            "transmitter_SNR": snr,
            "y_axis": 0,
            "x_start": 0
        })

    # 生成向左行驶的车 (direction = 1, y = road_width, x_start = road_length)
    for i in range(1, left_cars + 1):
        car_id = f"v{len(cars) + 1}"
        speed = random.choice(speeds_pool)
        snr = random.choice(snr_pool)
        cars.append({
            "car_ID": car_id,
            "speed": speed,
            "direction": 1,
            "transmitter_SNR": snr,
            "y_axis": road_width,
            "x_start": road_length
        })

    # 第四步：计算每个时间点的位置并写入 CSV
    filename = "sim_data_two_way.csv"
    headers = ["time", "car_ID", "x_axis", "y_axis", "speed", "direction", "transmitter_SNR"]

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for t in range(num_time_points):
                for car in cars:
                    # 计算 x 坐标
                    if car["direction"] == 0:
                        # 向右：x = speed * t
                        x_axis = car["speed"] * t
                    else:
                        # 向左：x = road_length - speed * t
                        x_axis = road_length - (car["speed"] * t)
                    
                    writer.writerow({
                        "time": t,
                        "car_ID": car["car_ID"],
                        "x_axis": x_axis,
                        "y_axis": car["y_axis"],
                        "speed": car["speed"],
                        "direction": car["direction"],
                        "transmitter_SNR": car["transmitter_SNR"]
                    })
        
        print(f"成功生成模拟数据并保存到 {filename}")
        print(f"总时间点数: {num_time_points}, 总数据行数: {num_time_points * number_cars}")

    except Exception as e:
        print(f"写入 CSV 文件时出错: {e}")

if __name__ == "__main__":
    # 测试用例：4辆车，200m长，10m宽，3右1左
    generate_car_sim_data(number_cars=20, road_length=5000, road_width=10, right_cars=10, left_cars=10)
