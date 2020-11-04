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
    max_steps = 20
    frame_idx = 0
    rewards = []
    x_axis1 = []
    var_a = 1.5

    while frame_idx < max_frames:
        state = [5, 0, 0]
        episode_reward = 0

        for step in range(max_steps):
            action = ddpg.policy_net.get_action(state)

            action[0] = action[0] + np.clip(np.random.normal(0, var_a), -1, 1)  # 添加噪声
            action[1] = action[1] + np.clip(np.random.normal(0, var_a), -1, 1)
            action[2] = action[2] + np.clip(np.random.normal(0, var_a), -1, 1)

            if frame_idx<=2:#固定theta
                reward, next_state, done , theta = environment(state, action, config)  # 输入环境中
            else:
                reward, next_state, done = environment2(state, action, theta, config)  # 输入环境中

            ddpg.replay_buffer.push(state, action, reward, next_state, done)

            if len(ddpg.replay_buffer) > config.BATCH_SIZE:
                var_a *= .9997
                ddpg.ddpg_update()
                episode_reward += reward

            state = next_state
            frame_idx += 1

            #每1000个epoch画一次图（调试用）
            if frame_idx % max(1000, max_steps + 1) == 0:
                plot(frame_idx, rewards)
                #print('Episode:', frame_idx, ' state:', state, ' action: ', action, ' Reward:', episode_reward, 'var_a', var_a, 'done',done)

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



