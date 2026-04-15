import csv
import math
import pandas as pd
import numpy as np
from fuzzy_logic import decide_use_nn

def compute_composite_snr_db(snr_trad_db, distance_m, rel_speed_ms, 
                             d_ref=1.0, d_min=1.0, 
                             n=2.0,            # path-loss exponent 
                             v_ref=10.0,       # reference speed (m/s) 
                             w_d=1.0, w_v=1.0, # weights for distance and speed penalties 
                             clip_min_db=-30.0, clip_max_db=40.0): 
    """ 
    Returns composite SNR in dB. 
    snr_trad_db: conventional SNR in dB (float or numpy scalar) 
    distance_m: distance in meters (float) 
    rel_speed_ms: relative speed in m/s (can be negative; function uses abs) 
    Other params are tunable. 
    """ 
    # sanitize inputs 
    d = max(distance_m, d_min) 
    v = abs(rel_speed_ms) 

    # distance penalty: 10 * n * log10(d / d_ref) scaled by w_d 
    pen_d_db = w_d * 10.0 * n * np.log10(d / d_ref) 

    # speed penalty: soft-log penalty, scaled by w_v 
    pen_v_db = w_v * 10.0 * np.log10(1.0 + (v / v_ref)) 

    composite = float(snr_trad_db) - float(pen_d_db) - float(pen_v_db) 

    # clip to avoid extreme values 
    composite = max(min(composite, clip_max_db), clip_min_db) 
    return composite 

def process_nearest_cars_data(start_time=10, end_time=None, input_file="sim_data_two_way.csv", output_file="nearest_cars_data.csv"):
    """
    在指定的 [start_time, end_time] 时间范围内，针对v1-v4一共4辆车，分别找到距离最近的5辆车，
    计算出相关的SNR、距离、相对速度、归一化数据、复合SNR和是否使用NN，并保存为CSV。
    """
    try:
        df_input = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return

    # 获取最大时间点
    max_time_in_data = df_input['time'].max()
    
    # 如果未指定 end_time，则默认为最大时间点
    if end_time is None:
        end_time = max_time_in_data

    # 校验参数
    if not (0 <= start_time < end_time <= max_time_in_data):
        print(f"错误：时间范围 [{start_time}, {end_time}] 不合法。")
        print(f"必须满足 0 <= start_time < end_time <= {max_time_in_data}。")
        return

    # 获取所有时间点并过滤出 [start_time, end_time] 的部分
    all_times = sorted(df_input['time'].unique())
    target_times = [t for t in all_times if start_time <= t <= end_time]
    
    # 目标 TX 车辆
    target_tx_cars = ["v1", "v2", "v3", "v4"]
    
    results = []

    for t in target_times:
        time_data = df_input[df_input['time'] == t]
        
        for tx_id in target_tx_cars:
            # 获取 TX 车辆信息
            tx_car = time_data[time_data['car_ID'] == tx_id]
            if tx_car.empty:
                continue
            
            tx_row = tx_car.iloc[0]
            tx_x, tx_y = tx_row['x_axis'], tx_row['y_axis']
            tx_speed = tx_row['speed']
            tx_dir = tx_row['direction']
            tx_snr = tx_row['transmitter_SNR']

            # 计算与其他车辆的距离
            other_cars = time_data[time_data['car_ID'] != tx_id].copy()
            if other_cars.empty:
                continue
            
            # 计算距离
            other_cars['dist'] = np.sqrt((other_cars['x_axis'] - tx_x)**2 + (other_cars['y_axis'] - tx_y)**2)
            
            # 找到最近的 5 辆车
            nearest_5 = other_cars.nsmallest(5, 'dist')
            
            for _, rx_row in nearest_5.iterrows():
                rx_id = rx_row['car_ID']
                rx_speed = rx_row['speed']
                rx_dir = rx_row['direction']
                dist = rx_row['dist']
                
                # 计算相对速度
                # 如果方向相同：|A - B|
                # 如果方向相反：|A - (-B)| = |A + B|
                if tx_dir == rx_dir:
                    rel_speed = abs(tx_speed - rx_speed)
                else:
                    rel_speed = abs(tx_speed + rx_speed)
                
                # 记录原始值
                res_row = {
                    "time": t,
                    "car_ID_TX": tx_id,
                    "car_ID_RX": rx_id,
                    "snr_values": float(tx_snr),
                    "distance_values": float(dist),
                    "rel_speed_values": float(rel_speed)
                }
                
                # 归一化计算
                # SNR: min = -10, max = 40
                res_row["snr_values_norm"] = (res_row["snr_values"] - (-10)) / (40 - (-10))
                # Distance: min = 0, max = 100
                res_row["distance_values_norm"] = (res_row["distance_values"] - 0) / (100 - 0)
                # Relative speed: min = 0, max = 50
                res_row["rel_speed_values_norm"] = (res_row["rel_speed_values"] - 0) / (50 - 0)
                
                # 计算复合 SNR
                res_row["composite_snr_db"] = compute_composite_snr_db(
                    res_row["snr_values"],
                    res_row["distance_values"],
                    res_row["rel_speed_values"]
                )
                
                # 模糊决策
                res_row["use_nn"] = decide_use_nn(
                    res_row["snr_values_norm"],
                    res_row["distance_values_norm"],
                    res_row["rel_speed_values_norm"]
                )
                
                results.append(res_row)

    # 转换为 DataFrame
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("未生成任何结果数据。")
        return

    # 设置精度
    # snr_values, distance_values, rel_speed_values 保留 2 位小数
    # snr_values_norm, distance_values_norm, rel_speed_values_norm, composite_snr_db 保留 4 位小数
    df_results["snr_values"] = df_results["snr_values"].round(2)
    df_results["distance_values"] = df_results["distance_values"].round(2)
    df_results["rel_speed_values"] = df_results["rel_speed_values"].round(2)
    
    df_results["snr_values_norm"] = df_results["snr_values_norm"].round(4)
    df_results["distance_values_norm"] = df_results["distance_values_norm"].round(4)
    df_results["rel_speed_values_norm"] = df_results["rel_speed_values_norm"].round(4)
    df_results["composite_snr_db"] = df_results["composite_snr_db"].round(4)
    
    # 保存为 CSV
    cols = ["time", "car_ID_TX", "car_ID_RX", "snr_values", "distance_values", "rel_speed_values", 
            "snr_values_norm", "distance_values_norm", "rel_speed_values_norm", "composite_snr_db", "use_nn"]
    df_results.to_csv(output_file, index=False, columns=cols)
    print(f"数据处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    # 示例：从时间点10开始到50结束
    process_nearest_cars_data(start_time=10, end_time=50)
