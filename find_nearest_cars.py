import csv
import math

def find_nearest_cars(time=5, car_ID="v2", adjacent_cars=1):
    """
    在指定时间点，找出距离目标车辆最近的车辆。
    
    参数:
    time (int): 指定的时间点
    car_ID (str): 目标车辆 ID (如 "v2")
    adjacent_cars (int): 需要找出的邻近车辆数量
    """
    filename = "sim_data_two_way.csv"
    data = []
    
    # 第一步：导入 CSV 文件
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                row['time'] = int(row['time'])
                row['x_axis'] = float(row['x_axis'])
                row['y_axis'] = float(row['y_axis'])
                data.append(row)
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
        return

    # 第二步：判断参数的有效性
    if not data:
        print("错误：CSV 文件为空")
        return

    # 获取所有时间点和车辆 ID
    all_times = sorted(list(set(row['time'] for row in data)))
    all_car_ids = sorted(list(set(row['car_ID'] for row in data)))
    total_cars = len(all_car_ids)
    max_time = all_times[-1]

    # time 参数校验
    if not (0 <= time <= max_time):
        print(f"错误：time 必须在 0 到 {max_time} 之间")
        return

    # car_ID 参数校验
    if car_ID not in all_car_ids:
        print(f"错误：car_ID {car_ID} 在数据中不存在")
        return

    # adjacent_cars 参数校验
    if not (1 <= adjacent_cars <= total_cars - 1):
        print(f"错误：adjacent_cars 必须在 1 到 {total_cars - 1} 之间")
        return

    # 第三步：计算距离
    # 过滤出指定时间点的所有车辆数据
    time_data = [row for row in data if row['time'] == time]
    
    # 找到目标车辆的位置
    target_car = next((row for row in time_data if row['car_ID'] == car_ID), None)
    if not target_car:
        print(f"错误：在 time={time} 时找不到车辆 {car_ID}")
        return

    target_x = target_car['x_axis']
    target_y = target_car['y_axis']

    # 计算其他车辆与目标车辆的距离
    distances = []
    for car in time_data:
        if car['car_ID'] == car_ID:
            continue
        
        dist = math.sqrt((car['x_axis'] - target_x)**2 + (car['y_axis'] - target_y)**2)
        distances.append((car['car_ID'], dist))

    # 按距离从小到大排序
    distances.sort(key=lambda x: x[1])

    # 打印结果
    for i in range(min(adjacent_cars, len(distances))):
        car_id, dist = distances[i]
        # 如果距离是整数，则显示整数，否则显示浮点数
        if dist == int(dist):
            print(f"{car_id}距离为{int(dist)}")
        else:
            print(f"{car_id}距离为{dist}")

if __name__ == "__main__":
    # 测试例子中的逻辑
    # 假设 CSV 中有对应的数据
    print("--- 运行测试 find_nearest_cars(time=0, car_ID='v2', adjacent_cars=3) ---")
    find_nearest_cars(time=51, car_ID="v2", adjacent_cars=3)
