### 2026-01-27   *1*
公路两端 发车，车辆ID，车辆间距，相对速度，距离-->SNR
设计一个从公路两端发车的模型，有车辆ID和车辆位置，车辆速度。由位置来计算距离，由距离来计算SNR。
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

### 2026-01-27   *2*
