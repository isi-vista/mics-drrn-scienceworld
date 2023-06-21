import time
import timeit
import torch
import logger
import argparse
from pathlib import Path
from drrn import DRRN_Agent
from vec_env import VecEnv

from scienceworld import ScienceWorldEnv, BufferedHistorySaver
from vec_env import resetWithVariation, resetWithVariationDev, resetWithVariationTest, \
    initializeEnv, sanitizeInfo, sanitizeObservation


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()


def evaluate(agent, args, env_step_limit, bufferedHistorySaverEval, extraSaveInfo):
    # Initialize a ScienceWorld thread for serial evaluation
    env = initializeEnv(threadNum=args.num_envs + 10,
                        args=args)  # A threadNum (and therefore port) that shouldn't be used by any of the regular training workers

    scoresOut = []
    if args.eval_set == "dev":
        eps = env.getVariationsDev()
    elif args.eval_set == "test":
        eps = env.getVariationsTest()
    else:
        env.shutdown()
        raise ValueError(f'`--eval_set` must be "dev" or "test", not "{args.eval_set}"')

    with torch.no_grad():
        for ep in eps:
            total_score = 0
            log("Starting evaluation episode {}".format(ep))
            print("Starting evaluation episode " + str(ep) + " / " + str(len(eps)))
            extraSaveInfo['evalIdx'] = ep
            score = evaluate_episode(agent, env, env_step_limit, args.simplification_str,
                                     bufferedHistorySaverEval, extraSaveInfo, args.eval_set, ep, args.save_actions)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
            scoresOut.append(total_score)
            print("")

        avg_score = sum(scoresOut) / len(scoresOut)

        # env.shutdown()

        return scoresOut, avg_score


def evaluate_episode(agent, env, env_step_limit, simplificationStr, bufferedHistorySaverEval,
                     extraSaveInfo, evalSet, variation, actions_save_dir):
    actions_save_file = actions_save_dir / f"{env.taskName}_{variation}.txt"
    actions = []
    step = 0
    done = False
    numSteps = 0
    env.load(taskName=env.taskName, variationIdx=variation, simplificationStr=simplificationStr)
    ob, info = env.reset()
    info = sanitizeInfo(info)
    ob = sanitizeObservation(ob, info)

    state = agent.build_state([ob], [info])[0]
    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']),
                                            clean(info['look'])))
    while not done:
        # print("numSteps: " + str(numSteps))
        valid_acts = info['valid']
        valid_ids = agent.encode(valid_acts)
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False)
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str,
                                                  action_values[action_idx].item()))
        s = ''

        maxToDisplay = 10  # Max Q values to display, to limit the log size
        numDisplayed = 0
        for idx, (act, val) in enumerate(
                sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
            numDisplayed += 1
            if (numDisplayed > maxToDisplay):
                break

        log('Q-Values: {}'.format(s))
        actions.append(action_str)
        ob, rew, done, info = env.step(action_str)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']),
                                                clean(info['look'])))
        state = agent.build_state([ob], [info])[0]

        numSteps += 1
        if (numSteps > env_step_limit):
            print("Maximum number of evaluation steps reached (" + str(env_step_limit) + ").")
            break

    print("Completed one evaluation episode")
    # Save
    actions_save_file.write_text("\n".join(actions))
    runHistory = env.getRunHistory()
    episodeIdx = str(extraSaveInfo['evalIdx'])
    bufferedHistorySaverEval.storeRunHistory(runHistory, episodeIdx, notes=extraSaveInfo)
    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(
        maxPerFile=extraSaveInfo['maxHistoriesPerFile'])
    print("Completed saving")

    return info['score'] if info['score'] >= 0 else 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--num_envs', default=16, type=int)
    parser.add_argument('--memory_size', default=5000000, type=int)
    parser.add_argument('--priority_fraction', default=0.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--task_idx', default=0, type=int)
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)
    parser.add_argument('--historySavePrefix', default='saveout', type=str)

    parser.add_argument('--eval_set', default='dev', type=str)  # 'dev' or 'test'

    parser.add_argument('--simplification_str', default='', type=str)

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--save_actions", default=None, type=Path)

    return parser.parse_args()


def main():
    ## assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    print(args)
    configure_logger(args.output_dir)
    agent = DRRN_Agent(args)
    agent.load(args.model_path, suffixStr=args.model_name)

    # Initialize the save buffers
    taskIdx = args.task_idx
    common_file_path = f"{args.output_dir}/{args.historySavePrefix}-task_{taskIdx}"
    output_file_eval = common_file_path + "-eval"
    print(f"Output file eval: {output_file_eval}")
    bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix=output_file_eval)

    # Start training
    start = timeit.default_timer()

    extraSaveInfo = {'maxHistoriesPerFile': args.maxHistoriesPerFile}
    eval_scores, avg_eval_score = evaluate(agent, args, args.env_step_limit,
                                           bufferedHistorySaverEval, extraSaveInfo)

    tb.logkv('EvalScore', avg_eval_score)
    tb.dumpkvs()

    for eval_score in eval_scores:
        print("EVAL EPISODE SCORE: " + str(eval_score))
    print("EVAL AVG SCORE: " + str(avg_eval_score))

    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)

    end = timeit.default_timer()
    deltaTime = end - start
    deltaTimeMins = deltaTime / 60
    print("Runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")

    print("SimplificationStr: " + str(args.simplification_str))


def interactive_run(env):
    ob, info = env.reset()
    while True:
        print(clean(ob), 'Reward', reward, 'Done', done, 'Valid', info)
        ob, reward, done, info = env.step(input())
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)


if __name__ == "__main__":
    main()
