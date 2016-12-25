#!/usr/bin/env python

from __future__ import print_function, division
import os
import random
import csv
import tensorflow as tf
import tflearn
import numpy as np
from envmaker import make_environment
from experiencebuffer import ExperienceBuffer
from dddqn_args import F

def get_num_actions():
    env = make_environment(F.game, F.height, F.width, F.num_channels)
    return env.get_num_actions()

def get_flat_state(state):
    return np.reshape(state, [F.num_channels * F.height * F.width])

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(max_reward=0.0)
def adjust_reward(reward):
    if F.reward_adjustment_method == "clip":
        adjusted_reward = np.clip(reward, -1.0, 1.0)
    elif F.reward_adjustment_method == "scale":
        adjust_reward.max_reward = max(reward, adjust_reward.max_reward)
        if adjust_reward.max_reward == 0.0:
            upper_clip = 1.0
        else:
            upper_clip = reward/adjust_reward.max_reward
        adjusted_reward = np.clip(reward, -1.0, upper_clip)
    elif F.reward_adjustment_method == "none":
        adjusted_reward = reward
    return adjusted_reward



def get_network_ops(nactions):
    # Environments states should have the shape of `reshaped_state`,
    # we then transpose it to have the shape of `inputs`. Note that
    # the network accepts flattened states which are initially in
    # in the shape of `reshaped_state`.
    # shape of reshaped_state = [batch, channel, height, width]
    # shape of inputs = [batch, height, width, channel]
    flat_state = tf.placeholder(tf.float32, [None, F.num_channels * F.height * F.width])
    reshaped_state = tf.reshape(flat_state, shape=[-1, F.num_channels, F.height, F.width])
    inputs = tf.transpose(reshaped_state, [0, 2, 3, 1])
    net = tflearn.conv_2d(inputs, 32, 8, strides=4, padding='valid', activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, padding='valid', activation='relu')
    net = tflearn.conv_2d(net, 64, 3, strides=1, padding='valid', activation='relu')
    net = tflearn.conv_2d(net, 512, 7, strides=1, padding='valid', activation='relu')

    advantage_net, value_net = tf.split(3, 2, net)
    advantage = tflearn.fully_connected(advantage_net, nactions)
    value = tflearn.fully_connected(value_net, 1)

    # See Eq. (9) in https://arxiv.org/pdf/1511.06581.pdf
    advantage = tf.sub(advantage,
                       tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
    q_values = value + advantage
    return flat_state, q_values

def get_graph_ops(nactions):
    current_state, online_q_values = get_network_ops(nactions)
    online_params = tf.trainable_variables()
    next_state, target_q_values = get_network_ops(nactions)
    target_params = tf.trainable_variables()[len(online_params):]

    update_target_params = \
        [target_params[i].assign(F.tau * online_params[i] \
                                         + (1-F.tau) * target_params[i])
         for i in range(len(target_params))]

    predict_action = tf.argmax(online_q_values, 1)
    target = tf.placeholder(shape=[None], dtype=tf.float32)
    action = tf.placeholder(shape=[None], dtype=tf.int32)
    onehot_action = tf.one_hot(action, nactions, dtype=tf.float32)
    acted = tf.reduce_sum(tf.mul(online_q_values, onehot_action), reduction_indices=1)
    td_error = tf.square(target - acted)
    loss = tf.reduce_mean(td_error)

    optimizers = {
        'adam': tf.train.AdamOptimizer(learning_rate=F.alpha),
        'rmsprop': tf.train.RMSPropOptimizer(learning_rate=F.alpha),
        'adadelta': tf.train.AdadeltaOptimizer(learning_rate=F.alpha),
        'adagrad': tf.train.AdadeltaOptimizer(learning_rate=F.alpha),
        'dradientdescent': tf.train.GradientDescentOptimizer(learning_rate=F.alpha)
    }

    trainer = optimizers[F.trainer]
    update_online_params = trainer.minimize(loss, var_list=online_params)

    ops = {'current_state': current_state,
           'online_q_values': online_q_values,
           'predict_action': predict_action,
           'target': target,
           'action': action,
           'update_online_params': update_online_params,
           'next_state': next_state,
           'target_q_values': target_q_values,
           'update_target_params': update_target_params}
    return ops


def get_summary_ops():
    summary_tags = ['Training Avrg Reward', 'Training Avrg Max Q',
                    'Validation Avrg Reward', 'Validation Avrg Max Q', 'Epsilon']
    summaries = {}
    summary_placeholders = {}
    for tag in summary_tags:
        summary_placeholders[tag] = tf.placeholder(shape=(), dtype=tf.float32)
        summaries[tag] = tf.scalar_summary(tag, summary_placeholders[tag])
    return summary_tags, summary_placeholders, summaries

def validate(session, graph_ops, env):
    op_current_state = graph_ops['current_state']
    op_online_q_values = graph_ops['online_q_values']

    avrg_reward = 0.0
    avrg_max_q = 0.0
    for _ in range(F.num_validation_episodes):
        state = env.reset()
        state = get_flat_state(state)
        ep_reward = 0.0
        ep_max_q = 0.0
        ep_step = 0
        done = False
        while not done:
            ep_step += 1
            online_q_values = session.run(op_online_q_values,
                                          feed_dict={op_current_state: [state]})
            action = np.argmax(online_q_values)
            state, reward, done, _ = env.step(action)
            state = get_flat_state(state)
            ep_max_q += np.max(online_q_values)
            ep_reward += reward

        ep_max_q /= ep_step
        avrg_reward += ep_reward
        avrg_max_q += ep_max_q

    avrg_reward /= F.num_validation_episodes
    avrg_max_q /= F.num_validation_episodes
    return avrg_reward, avrg_max_q



def train(session, graph_ops, nactions, saver):
    if F.checkpoint_path:
        print('Loading pre-trained model', F.checkpoint_path)
        saver.restore(session, F.checkpoint_path)

    session.run(tf.initialize_all_variables())
    summary_save_path = F.summary_dir + "/" + F.experiment
    writer = tf.train.SummaryWriter(summary_save_path, session.graph)
    if not os.path.exists(F.checkpoint_dir):
        os.makedirs(F.checkpoint_dir)

    csv_file = open(summary_save_path + "/plot.csv", "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(("step", "episode", "validation_reward",
                         "validation_max_q", "train_reward", "train_max_q", "epsilon"))
    csv_file.flush()

    op_current_state = graph_ops['current_state']
    op_online_q_values = graph_ops['online_q_values']
    op_predict_action = graph_ops['predict_action']
    op_target = graph_ops['target']
    op_action = graph_ops['action']
    op_update_online_params = graph_ops['update_online_params']
    op_next_state = graph_ops['next_state']
    op_target_q_values = graph_ops['target_q_values']
    op_update_target_params = graph_ops['update_target_params']
    session.run(op_update_target_params)

    summary_tags, op_summary_placeholders, op_summaries = get_summary_ops()

    env = make_environment(F.game, F.width, F.height, F.num_channels)
    step = 0
    drop_epsilon = (F.start_epsilon - F.final_epsilon) / F.epsilon_annealing_episodes
    epsilon = F.start_epsilon
    ex_buffer = ExperienceBuffer(F.experience_buffer_size)
    avrg_reward = 0.0
    avrg_max_q = 0.0
    for ep_counter in range(F.num_training_episodes + 1):
        ep_buffer = ExperienceBuffer(F.experience_buffer_size)
        ep_step = 0
        ep_reward = 0.0
        ep_avrg_max_q = 0.0
        state = env.reset()
        current_state = get_flat_state(state)
        while True:
            step += 1
            ep_step += 1
            action_array, online_q_values = session.run([op_predict_action, op_online_q_values],
                                    feed_dict={op_current_state: [current_state]})
            ep_avrg_max_q += np.max(online_q_values)
            action = action_array[0]
            if random.random() < epsilon or ep_counter < F.pre_training_episodes:
                action = random.randrange(nactions)

            state, reward, done, _ = env.step(action)
            next_state = get_flat_state(state)
            adjusted_reward = adjust_reward(reward)
            ep_buffer.add(np.reshape(
                np.array([current_state, action, adjusted_reward, next_state, done]), [1, 5]))

            if ep_counter > F.pre_training_episodes:
                if step % F.update_frequency == 0:
                    batch = ex_buffer.sample(F.batch_size)
                    next_state_batch = np.vstack(batch[:, 3])
                    current_state_batch = np.vstack(batch[:, 0])
                    action_batch = batch[:, 1]
                    actions = session.run(op_predict_action,
                                        feed_dict={op_current_state: next_state_batch})
                    target_q_values = session.run(op_target_q_values,
                                                feed_dict={op_next_state: next_state_batch})
                    double_q_values = target_q_values[range(F.batch_size), actions]
                    not_done = -(batch[:, 4] - 1)
                    target = batch[:, 2] + (F.gamma * double_q_values * not_done)
                    session.run(op_update_online_params,
                                feed_dict={op_current_state: current_state_batch,
                                           op_target: target,
                                           op_action: action_batch})
                    session.run(op_update_target_params)

            ep_reward += reward
            current_state = next_state
            if done:
                break

        if ep_counter > F.pre_training_episodes and epsilon > F.final_epsilon:
            epsilon -= drop_epsilon

        ep_avrg_max_q /= ep_step
        avrg_reward += ep_reward
        avrg_max_q += ep_avrg_max_q
        if ep_counter % F.summary_interval == 0:
            avrg_reward /= F.summary_interval
            avrg_max_q /= F.summary_interval
            validation_avrg_reward, validation_avrg_max_q = validate(session, graph_ops, env)
            stats = [avrg_reward, avrg_max_q, validation_avrg_reward, validation_avrg_max_q, epsilon]
            tag_dict = {}
            for index, tag in enumerate(summary_tags):
                tag_dict[tag] = stats[index]

            summary_str_lists = session.run([op_summaries[tag] for tag in tag_dict.keys()],
                feed_dict={op_summary_placeholders[tag]: value for tag, value in tag_dict.items()})

            for summary_str in summary_str_lists:
                writer.add_summary(summary_str, ep_counter)

            fmt = "STEP {:8d} | EPISODE {:6d} | AVRG_REWARD {:.2f} | AVRG_MAX_Q {:.4f} | EPSILON {:.4f}"
            print(fmt.format(step, ep_counter, stats[2], stats[3], stats[4]))

            csv_writer.writerow((step, ep_counter, validation_avrg_reward, validation_avrg_max_q,
                                 avrg_reward, avrg_max_q, epsilon))
            csv_file.flush()
            avrg_reward = 0.0
            avrg_max_q = 0.0


        ex_buffer.add(ep_buffer.buff)
        if ep_counter % F.checkpoint_interval == 0:
            saver.save(session, F.checkpoint_dir + "/"
                        + F.experiment + ".ckpt", global_step=ep_counter)
    csv_file.close()

def test(session, graph_ops, saver):
    print('Loading pre-trained model', F.checkpoint_path)
    saver.restore(session, F.checkpoint_path)
    env = make_environment(F.game, F.width, F.height, F.num_channels)
    env.monitor_start(F.eval_dir)

    op_current_state = graph_ops['current_state']
    op_online_q_values = graph_ops['online_q_values']

    for _ in range(F.num_testing_episodes):
        state = env.reset()
        state = get_flat_state(state)
        ep_reward = 0.0
        done = False
        while not done:
            env.render()
            online_q_values = session.run(op_online_q_values,
                                          feed_dict={op_current_state: [state]})
            action = np.argmax(online_q_values)
            state, reward, done, _ = env.step(action)
            state = get_flat_state(state)
            ep_reward += reward
        print("EPISODE REWARD:", ep_reward)
    env.monitor_close()

if __name__ == "__main__":
    with tf.Session() as session:
        nactions = get_num_actions()
        graph_ops = get_graph_ops(nactions)
        saver = tf.train.Saver()

        if F.subcommand == "test":
            test(session, graph_ops, saver)
        elif F.subcommand == "train":
            train(session, graph_ops, nactions, saver)
