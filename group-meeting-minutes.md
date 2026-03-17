### 2026-01-27 - *1*
公路两端 发车，车辆ID，车辆间距，相对速度，距离-->SNR
设计一个从公路两端发车的模型，有车辆ID和车辆位置，车辆速度。由位置来计算距离，由距离来计算SNR。

请帮我用 Python 做一个车辆模拟程序，并生成一个名为 sim_data.csv 的文件。
假设有 A、B、C、D、E 五辆车从一条直路上出发。
这五辆车的速度分别为 5、10、15、20 和 25 米/秒。
也就是说，A 车速为 5 米/秒，B 车速为 10 米/秒，C 车速为 15 米/秒，D 车速为 20 米/秒，E 车速为 25 米/秒。
车辆行驶了 5 秒。
每辆车的位置如下表所示，单位为米。

时间 A B C D E
1秒 5 10 15 20 25
2秒 10 20 30 40 50
3秒 15 30 45 60 75
4秒 20 40 60 80 100
5秒 25 50 75 100 125

请按照以下说明生成数据。
在A车和B车之间生成了5行数据。
在A车和C车之间生成了5行数据。
在A车和D车之间生成了5行数据。
在A车和E车之间生成了5行数据。
总共生成了20行数据。

假设A车发射器的信噪比为40dB。信号​​强度随距离衰减。
请生成信噪比 (SNR)、距离和相对速度值，然后将数据归一化到 0 到 1 的范围内。

我假设归一化的上下限如下：
信噪比：最大值 40，最小值 -10
距离：最大值 100，最小值 0
相对速度：最大值 25，最小值 0

总之，sim_data.csv 文件包含 6 列：
snr_values、distance_values、rel_speed_values、snr_values_norm、distance_values_norm 和 rel_speed_values_norm
sim_data.csv 文件包含 20 行数据。

please help me make a simulation of vehicles with Python to generate a file sim_data.csv
assuming that there are A B C D E five cars to start from a straight road.
The speeds of the 5 cars are 5 10 15 20 25 m/s respectively.
That is to say, A 5 m/s, B 10 m/s, C 15 m/s, D 20 m/s, E 25 m/s,
The cars traveled for 5 seconds.
The position of each vehicle is shown in the table below, in meters.

time	A	B	C	D	E
1s	5	10	15	20	25
2s	10	20	30	40	50
3s	15	30	45	60	75
4s	20	40	60	80	100
5s	25	50	75	100	125

please generate data as the follow instructions.
5 rows of data were generated between car A and car B.
5 rows of data were generated between car A and car C.
5 rows of data were generated between car A and car D.
5 rows of data were generated between car A and car E.
A total of 20 rows of data.
assuming the SNR value of transmitter on car A is 40dB. The signal weakens with distance.
please generate SNR, distance and relative speed values, and then normalize the data to a range of 0 to 1.
I assume the normalization upper and lower bounds are as follows.
for SNR, max is 40, min is -10
for distance, max is 100, min is 0
for relative speed, max is 25, min is 0

In conclusion, there are 6 columns in sim_data.csv, 
snr_values, distance_values, rel_speed_values, snr_values_norm, distance_values_norm, rel_speed_values_norm
there are 20 rows of data in sim_data.csv

please give me the python code, so that I can run locally

### 2026-03-11 - *2*
TVT TNSE TCOM TCCN IOT journal TITS TII JSAC

### 2026-03-18 - *3*
