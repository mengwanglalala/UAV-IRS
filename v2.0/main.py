import argparse
from shutil import copyfile
from lib.config import Config
from envs import *
from network import *
import os
import pickle

def main():
    config = load_config()

    action_dim = 3
    state_dim = 3
    hidden_dim = 64  # 不易过多，因为输出输出都为3，过多会导致系统不稳定

    ddpg = DDPG(action_dim, state_dim, hidden_dim, config)

    max_frames = 15000
    max_steps = 30
    frame_idx = 0
    rewards = []
    x_axis1 = []
    var_a = 1.5

    while frame_idx < max_frames:
        state = [5,0,0]
        statex = []
        statey = []
        episode_reward = 0
        F = [0,0,0]

        for step in range(max_steps):
            action = ddpg.policy_net.get_action(state)

            action[0] = action[0] + np.clip(np.random.normal(0, var_a), -1, 1)  # 添加噪声
            action[1] = action[1] + np.clip(np.random.normal(0, var_a), -1, 1)
            action[2] = action[2] + np.clip(np.random.normal(0, var_a), -1, 1)

            if frame_idx<=2:#固定theta
                reward, next_state, done , theta = environment(state, action, config)  # 输入环境中
            else:
                reward, next_state, done , sensori = environment2(state, action, theta, config)  # 输入环境中

                #加入公平系数
                F[sensori] = F[sensori] + 1
                F_all = F[0] + F[1] + F[2]
                F_t = (F[0] + F[1] + F[2]) ** 2 / (3 * (F[0] ** 2 + F[1] ** 2 + F[2] ** 2))
                # if reward>0:
                #     reward = F_t * reward

            ddpg.replay_buffer.push(state, action, reward, next_state, done)

            if len(ddpg.replay_buffer) > config.BATCH_SIZE:
                var_a *= .9997
                ddpg.ddpg_update()
                episode_reward += reward

            statex.append(state[0])
            statey.append(state[1])
            state = next_state

            frame_idx += 1

            #每1000个epoch画一次图（调试用）
            if frame_idx % max(1000, max_steps + 1) == 0:
                plot(frame_idx, rewards)
                n = 4
                x = [0, 13, 10, 12]
                y = [0, -3, -5, 0]
                x = np.array(x)
                y = np.array(y)
                fig = plt.figure(1)
                # colors = np.random.rand(n) # 随机产生10个0~1之间的颜色值，或者
                colors = ['r', 'g', 'g', 'g']  # 可设置随机数取
                area = [300, 100, 100, 100]
                widths = np.arange(n)  # 0-9的数字
                plt.scatter(statex, statey, s=1, c='b', linewidths=5, alpha=0.5, marker='o')
                plt.scatter(x, y, s=area, c=colors, linewidths=widths, alpha=0.5, marker='o')
                # 设置X轴标签
                plt.xlabel('X坐标')
                # 设置Y轴标签
                plt.ylabel('Y坐标')
                plt.title('无人机路径优化')
                # 设置横轴的上下限值
                plt.xlim(-5, 15)
                # 设置纵轴的上下限值
                plt.ylim(-10, 10)
                # 设置横轴精准刻度
                plt.xticks(np.arange(-5, 15, step=1))
                # 设置纵轴精准刻度
                plt.yticks(np.arange(-10, 10, step=1))
                # plt.annotate("({0},{1})".format(round(x[0], 2), round(y[0], 2)), xy=(x[0], y[0]), fontsize=10, xycoords='data')
                # plt.text(round(x[0], 2), round(y[0], 2), "UAV",fontdict={'size': 10, 'color': 'blue'})  # fontdict设置文本字体
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

                plt.legend(['UAV轨迹'], loc=2, fontsize=10)
                # plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
                plt.show()

            if done ==1 :
                break




        rewards.append(episode_reward)
        print('Episode:', frame_idx, ' state:', state, ' action: ', action, ' Reward:', episode_reward, 'var_a', var_a, 'done',done)
        x_axis1.append(frame_idx)

        if done == 2:  # 扔掉出现意外的点并记录
            mylog = open('cvx_error.log', mode='a', encoding='utf-8')
            print('!!!error!!!', 'Episode:', frame_idx, ' state:', state, ' action: ', action,
                  ' Reward:', episode_reward, 'var_a', var_a)
            mylog.close()

    plt.figure(1)
    plt.plot(x_axis1, rewards, linestyle='-.')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.legend('Optimization-driven DDPG', loc='lower right')
    plt.show()

    # 存入reward
    with open('./checkpoints/x_reward.pkl', 'wb') as f:
        pickle.dump(x_axis1, f)
    with open('./checkpoints/y_reward.pkl', 'wb') as f:
        pickle.dump(rewards, f)


def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)
    config.print()

    return config


if __name__ == '__main__':
    main()



