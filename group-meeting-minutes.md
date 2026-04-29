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
增加会车情况

### 2026-03-25 - *4*
DOI: 10.1109/TVT.2024.3521948
修改generate_sim_data_two_way.py里的函数generate_car_sim_data
第一，把road_width的限制条件改为大于等于10，且为10的整数倍。
第二，把number_cars改为必须大于等于road_width/10取整数再加1，即如果road_width=10，number_cars>=10/10+1，即number_cars大于2，以确保每条车道最少有1辆车。
第三，把车辆平均分配到间距为10的车道（纵坐标Y）上，y坐标小于等于中值（road_width/2）的车道车辆向右行驶（x_start = 0），y坐标大于中值（road_width/2）的车道车辆向左行驶（x_start = road_length），例如road_width=20，road_width/2=10即为车道0（y = 0）,10（y = 10）车辆向右，车道20（y = 20）车辆向左，再假设number_cars=4，把
v1,v2,v3,v4用循环地方式分配到3个车道上，v1车道0,v2车道10,v3车道20,v4车道0。
第四，把车辆的速度由匀速改为变速，即每个时间段内的速度改为随机值random.choice(speeds_pool)，去掉num_time_points时间限制，把计算位置循环的结束条件改为最后一辆车驶出车道，即向右车辆的x_axis从0开始递增，向左车辆的x_axis从road_length开始递减。直到向右车辆的x_axis从0开始递增到全部车辆x_axis>=road_length，并且向左车辆的x_axis从road_length开始递减到全部车辆x_axis<=0时，退出循环，此时所有车辆驶出车道。

### 2026-04-01 - *5*
改进车道宽度
### 2026-04-08 - *6*
修改发车间隔
### 2026-04-15 - *7*
1. 模糊逻辑决策可视化

生成了用于自适应传输的模糊决策系统的可视化图形。

2. 综合信噪比（SNR）建模与可视化

利用 `compute_composite_snr_db()` 函数对信道模型进行了分析。

该模型综合考量了以下因素：
距离衰减（对数距离路径损耗）
相对速度惩罚（移动性效应）

3. 自适应语义通信实验（CIFAR-10）

设计了一套完整的仿真流程，用于评估自适应传输策略。

实验设置
数据集：CIFAR-10（测试集）
模型：5个预训练的语义编码器，分别在不同的信噪比（SNR）条件下进行训练：
10 dB、15 dB、17.5 dB、20 dB、25 dB
信道数据：
通过车辆仿真生成（`sim_data_two_way.csv`）
经处理后提取为“最近车辆对”数据（`nearest_cars_data.csv`）
传输策略

对于每一个通信实例：

利用模糊逻辑进行决策：
语义传输（通过神经网络编码器）
直接传输（原始数据 + 噪声）
根据综合信噪比（SNR）添加相应的噪声
评估指标
准确率（Accuracy）：基于预训练的 GoogleNet 进行分类评估
峰值信噪比（PSNR）：衡量重构质量
结果输出

实验结果保存至：
`./results/transmission_cifar/transmission_cifar_data.csv`
每一行数据包含：
信道条件（SNR、距离、速度）
决策结果（语义传输 / 直接传输）
所使用的模型
准确率与 PSNR 数值

#### original manuscript
Task 1, please write a python code for me to generate a schematic diagram for the result of fuzzy_logic.py which I uploaded. 
1. generate 3 graphs, SNR is low (snr_norm=0), SNR is medium (snr_norm=0.5), SNR is high (snr_norm=1)
2. the speed and distance are the x-axis and y-axis of each graph, use 0.25 as the axis label marking gap, .e.g 0 0.25 0.5 0.75 1.0
3. divide the chart into small squares one by one, then paint them. semantic is red, direct is blue.
4. the graphs should look like below tables. 

SNR	low		
speed\distance	near	medium	far
slow	semantic	semantic	semantic
medium	semantic	semantic	semantic
fast	semantic	semantic	semantic
SNR	medium		
speed\distance	near	medium	far
slow	direct	semantic	semantic
medium	semantic	semantic	semantic
fast	semantic	semantic	semantic
SNR	high		
speed\distance	near	medium	far
slow	direct	direct	semantic
medium	direct	direct	semantic
fast	semantic	semantic	semantic

Task 2, please help me draw 3 graphs for the function compute_composite_snr_db() in CIFAR.py
1. we suppose snr_trad_db=40, please give me the python code to draw a 3D graph in which the speed and distance are the x-axis and y-axis, the distance range is between 0m and 100m, the relative speed range is between 0m/s and 50m/s. 
2. generate the code to draw a 2D graph where snr_trad_db=40, distance_m=100, composite SNR decreases with speed.
3. generate the code to draw a 2D graph where snr_trad_db=40, rel_speed_ms=50, composite SNR decreases with distance.

Task 3, design experiment for adaptive semantic communication of image transmission.
please design and generate code based on the training code in CIFAR.py
please help me to design an experiment for adaptive semantic communication. 
I want to simulate a transmission of images in cifar-10 dataset. please test it based on the test set.
1. you can load the trained models from ./saved_models folder, for example, saved_models\CIFAR_encoder_1.000000_snr_10.00.pkl, I want you to test 5 models, namely 5 loops.
CIFAR_encoder_1.000000_snr_10.00.pkl
CIFAR_encoder_1.000000_snr_15.00.pkl
CIFAR_encoder_1.000000_snr_17.50.pkl
CIFAR_encoder_1.000000_snr_20.00.pkl
CIFAR_encoder_1.000000_snr_25.00.pkl
2. transmit the images through 2 different ways, using fuzzy_logic.py to decide when to use direct or semantic models.
2.1 When go through direct channel, you add noise to the transmission process based on composite SNR which is computed in the function compute_composite_snr_db(). Please calculate the noise according to composite SNR.
2.2 When go through semantic channel, you put the image through the trained models from ./saved_models folder.

3. I generated car simulation data by generate_sim_data_two_way.py which created this file sim_data_two_way.csv.
Then I generate nearest_cars_data.csv using the code find_nearest_cars.py. Please simulate the transmission using the data from nearest_cars_data.csv, below is the front 30 rows of the data.
time,car_ID_TX,car_ID_RX,snr_values,distance_values,rel_speed_values,snr_values_norm,distance_values_norm,rel_speed_values_norm,composite_snr_db,use_nn
10,v1,v2,40.0,20.22,20.0,1.0,0.2022,0.4,9.1116,False
10,v1,v5,40.0,110.0,20.0,1.0,1.1,0.4,-5.5991,True
10,v1,v6,40.0,190.02,10.0,1.0,1.9002,0.2,-8.5865,True
10,v1,v9,40.0,260.0,10.0,1.0,2.6,0.2,-11.3098,True
10,v1,v10,40.0,280.02,0.0,1.0,2.8002,0.0,-8.9437,True

4. please calculate the accuracy and PSNR in each epoch, then save it into the folder ./results/transmission_cifar

In conclusion, the experiment should look like below:
  1st loop, 5 models
    2dn loop, every line in the nearest_cars_data.csv
	  if use_nn then
	    3rd loop, loop every image of test set in cifar-10 dataset, transmit every image through semantic models.
	  else
	    3rd loop, loop every image of test set in cifar-10 dataset, transmit every image through direct with noise channel.
	  add data to file ./results/transmission_cifar/transmission_cifar_data.csv, first copy all the columns in nearest_cars_data.csv, then add the calculated accuracy and psnr data to the file, that is to say to append column "accuracy" and column "psnr" to transmission_cifar_data.csv

### 2026-04-22 - *8*
完全语义
direct
最近，random，相对速度最慢。

### 2026-04-29 - *9*
进行了消融实验，绘制了use_nn_true和use_nn_false两种状态下的数据散点图和折线图，对比了完全进行语义通信和完全进行有复合噪声的直接通信，目前实验结果不太理想。计划下一步继续改进实验流程，更改实验代码架构，进行encoder和decoder的联合训练，分开部署。
