# UAV-IRS
UAV－IRS
无人机携带IRS服务地面用户辅助通信任务，无人机轨迹优化使用DRL方法解决，IRS采用CVX包进行优化求解

V0版本
matlab与python混合编程（但是训练速度很慢）
利用python对matlab的所有程序进行复现

V1版本（更新中）
完全替换掉了matlab，针对单用户的优化（最优位置在用户处）
1.1版本达到了DDPG优化的效果，之后版本会增加考虑一些限制
1.2版本基本解决1.1版本出现的reward骤降的bug

V2版本（待更新）
针对多用户的情况
2.0版本加入多用户,无人机在任意时刻只服务于一个用户,最大化此snr.
引入公平系数防止无人机一直优化至某一用户正上方,使用的是固定theta,在初始位置优化了三个优化的theta.
