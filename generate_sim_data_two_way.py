import csv
import random

def generate_car_sim_data(number_cars=20, road_length=5000, road_width=10):
    """
    生成车辆行驶模拟数据并保存为 CSV 文件。
    
    参数:
    number_cars (int): 车辆总数 (>= road_width/10 + 1)
    road_length (int): 道路长度 (>= 100 且为 100 的倍数)
    road_width (int): 道路宽度 (>= 10 且为 10 的倍数)
    """
    
    # 第一步：检查输入参数
    if road_width < 10 or road_width % 10 != 0:
        print("错误：road_width 必须大于等于 10，且为 10 的整数倍")
        return

    min_cars = int(road_width / 10) + 1
    if number_cars < min_cars:
        print(f"错误：number_cars 必须大于等于 {min_cars} (road_width/10 + 1)")
        return

    if road_length < 100 or road_length % 100 != 0:
        print("错误：road_length 必须大于等于 100 且为 100 的倍数")
        return

    # 第三步：生成车辆初始属性
    cars = []
    speeds_pool = [10, 20, 30, 40, 50]
    snr_pool = [20, 30, 40]
    random.seed(42)

    # 车道分配 (间距为 10)
    lanes = list(range(0, road_width + 1, 10))
    mid_road = road_width / 2

    for i in range(number_cars):
        car_id = f"v{i + 1}"
        y_axis = lanes[i % len(lanes)]
        
        # y 坐标 <= 中值向右 (direction=0), y 坐标 > 中值向左 (direction=1)
        if y_axis <= mid_road:
            direction = 0
            x_axis = 0
        else:
            direction = 1
            x_axis = road_length
            
        snr = random.choice(snr_pool)
        cars.append({
            "car_ID": car_id,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "direction": direction,
            "transmitter_SNR": snr,
            "finished": False
        })

    # 第四步：计算位置并写入 CSV
    filename = "sim_data_two_way.csv"
    headers = ["time", "car_ID", "x_axis", "y_axis", "speed", "direction", "transmitter_SNR"]

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            t = 0
            while True:
                active_cars = [car for car in cars if not car['finished']]
                if not active_cars:
                    break

                for car in cars:
                    if car['finished']:
                        speed = 0  # 如果车辆已经完成行驶，速度为0
                        writer.writerow({
                            "time": t,
                            "car_ID": car["car_ID"],
                            "x_axis": car["x_axis"],
                            "y_axis": car["y_axis"],
                            "speed": speed,
                            "direction": car["direction"],
                            "transmitter_SNR": car["transmitter_SNR"]
                        })
                    if not car['finished']:
                        # 随机选择当前时间段的速度
                        speed = random.choice(speeds_pool)
                        
                        # 写入当前时间点的数据 (位置是本秒初的位置)
                        writer.writerow({
                            "time": t,
                            "car_ID": car["car_ID"],
                            "x_axis": car["x_axis"],
                            "y_axis": car["y_axis"],
                            "speed": speed,
                            "direction": car["direction"],
                            "transmitter_SNR": car["transmitter_SNR"]
                        })

                        # 更新位置供下一秒使用
                        if car["direction"] == 0:
                            car["x_axis"] += speed
                            if car["x_axis"] >= road_length:
                                car["finished"] = True
                        else:
                            car["x_axis"] -= speed
                            if car["x_axis"] <= 0:
                                car["finished"] = True
                t += 1
        
        print(f"成功生成模拟数据并保存到 {filename}")
        print(f"总时间点数: {t}")

    except Exception as e:
        print(f"写入 CSV 文件时出错: {e}")

if __name__ == "__main__":
    # 测试用例：4辆车，200m长，20m宽
    generate_car_sim_data(number_cars=4, road_length=200, road_width=20)

