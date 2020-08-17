from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
from env import make_env
from controller import make_controller, simulate
from es import CMAES, SimpleGA, OpenES, PEPG
from utils import PARSER
import argparse
import time

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
###

def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
  global population, filebase, game, controller, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
  population = num_worker * num_worker_trial
  filedir = 'results/{}/{}/log/'.format(exp_name, env_name)
  if not os.path.exists(filedir):
      os.makedirs(filedir)
  filebase = filedir+env_name+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  controller = make_controller(args=config_args)

  num_params = controller.param_count
  print("size of model", num_params)

  if optimizer == 'ses':
    ses = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.2,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ses
  elif optimizer == 'ga':
    ga = SimpleGA(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ga
  elif optimizer == 'cma':
    cma = CMAES(num_params,
      sigma_init=sigma_init,
      popsize=population)
    es = cma
  elif optimizer == 'pepg':
    pepg = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.20,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      weight_decay=0.005,
      popsize=population)
    es = pepg
  else:
    oes = OpenES(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      antithetic=antithetic,
      weight_decay=0.005,
      popsize=population)
    es = oes

  PRECISION = 10000
  SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
  RESULT_PACKET_SIZE = 2*num_worker_trial + 2 * num_episode * num_worker_trial # worker and job id for each worker + return list and timestep list for each worker
###

def sprint(*args):
  print(args) # if python3, can do print(*args)
  sys.stdout.flush()

class OldSeeder:
  def __init__(self, init_seed=0):
    self._seed = init_seed
  def next_seed(self):
    result = self._seed
    self._seed += 1
    return result
  def next_batch(self, batch_size):
    result = np.arange(self._seed, self._seed+batch_size).tolist()
    self._seed += batch_size
    return result

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
  n = len(seeds)
  result = []
  worker_num = 0
  for i in range(n):
    worker_num = int(i / num_worker_trial) + 1
    result.append([worker_num, i, seeds[i], train_mode, max_len])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))
  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)
  return result

def decode_solution_packet(packet):
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float)/PRECISION])
  return result

def encode_result_packet(results):
  r = np.reshape(np.array(results), [-1,])
  r = np.concatenate([np.array(A).flatten() for A in r], axis=0)
  eval_packet_size = 2*num_worker_trial + 2 * num_test_episode * num_worker_trial # not the same size for training
  r[2:] *= PRECISION
  if r.size == eval_packet_size:
      r = np.concatenate([r, np.zeros(RESULT_PACKET_SIZE - eval_packet_size)-1.0], axis=0)
  return r.astype(np.int32)

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, -1)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  fits = r[:, 2:(2+num_worker_trial*num_episode)].astype(np.float)/PRECISION
  fits = fits.tolist()
  times = r[:, (2+num_worker_trial*num_episode):].astype(np.float)/PRECISION
  times = times.tolist()
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], fits[i], times[i]])
  return result

def worker(weights, seed, train_mode_int=1, max_len=-1):
  train_mode = (train_mode_int == 1) # perfomring training if 1. performing evaluation if -1 or 0
  controller.set_model_params(weights)
  if train_mode_int == 1: # train with DREAM env
    env._training = train_mode
    reward_list, t_list = simulate(controller, env,
	    train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
  elif train_mode_int == 0: # eval with DREAM env
    env._training = True #train_mode
    reward_list, t_list = simulate(controller, env,
        train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
  elif train_mode_int == -1: # eval with REAL env
    reward_list, t_list = simulate(controller, test_env,
        train_mode=train_mode, render_mode=False, num_episode=num_test_episode, seed=seed, max_len=max_len)
  return reward_list, t_list

def slave():
  global env, test_env
  env = make_env(args=config_args, dream_env=config_args.dream_env)
  # doom env doesn't support mpi testing so don't bother loading
  if 'DoomTakeCover-v0' != config_args.env_name:
    test_env = make_env(args=config_args, dream_env=False, render_mode=False)

  packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
  while 1:
    comm.Recv(packet, source=0)
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    solutions = decode_solution_packet(packet)
    results = []
    for solution in solutions:
      worker_id, jobidx, seed, train_mode, max_len, weights = solution
      assert (train_mode == 1 or train_mode == 0 or train_mode == -1), str(train_mode)
      worker_id = int(worker_id)
      possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
      assert worker_id == rank, possible_error
      jobidx = int(jobidx)
      seed = int(seed)
      fitness, timesteps = worker(weights, seed, train_mode, max_len)
      results.append([worker_id, jobidx, fitness, timesteps])
    result_packet = encode_result_packet(results)
    assert len(result_packet) == RESULT_PACKET_SIZE
    comm.Send(result_packet, dest=0)

def send_packets_to_slaves(packet_list):
  num_worker = comm.Get_size()
  assert len(packet_list) == num_worker-1
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    comm.Send(packet, dest=i)

def receive_packets_from_slaves():
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

  reward_list_total = np.zeros((population, 2 * num_episode))

  check_results = np.ones(population, dtype=np.int)
  for i in range(1, num_worker+1):
    comm.Recv(result_packet, source=i)
    results = decode_result_packet(result_packet)
    for result in results:
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, :num_episode] = result[2]
      reward_list_total[idx, num_episode:] = result[3]
      check_results[idx] = 0

  check_sum = check_results.sum()
  assert check_sum == 0, check_sum
  return reward_list_total
	
def evaluate_batch(model_params, train_mode, max_len=-1):
  # runs only from master since mpi and Doom was janky
  if args.env_name == 'DoomTakeCover-v0' and train_mode==-1: # can't run the real environment in parallel for doom
    controller.set_model_params(model_params)
    rewards_list, t_list = simulate(controller, test_env,
          train_mode=train_mode, render_mode=False, num_episode=100, seed=0, max_len=max_len) # run exactly 100 episodes b/c we cant run in parallel
  else:
    # duplicate model_params
    solutions = []
    for i in range(es.popsize):
      solutions.append(np.copy(model_params))
    seeds = np.arange(es.popsize)
    packet_list = encode_solution_packets(seeds, solutions, train_mode=train_mode, max_len=max_len)
    send_packets_to_slaves(packet_list)
    response_list_total = receive_packets_from_slaves()

    if train_mode == -1: # evaluate on 100 episodes in the TRUE environment
        # num test episode is defined per agent. So we receive popsize * num_test_episode episodes
        # report results with 100 episodes
        rewards_list = response_list_total[:, :(num_test_episode)].flatten()[:100] # get rewards
    else:
        rewards_list = response_list_total[:, :num_episode].flatten() # get rewards
  return rewards_list

def master():
  global test_env
  test_env = make_env(args=config_args, dream_env=False, render_mode=False)

  start_time = int(time.time())
  sprint("training", env_name)
  sprint("population", es.popsize)
  sprint("num_worker", num_worker)
  sprint("num_worker_trial", num_worker_trial)
  sys.stdout.flush()

  seeder = Seeder(seed_start)

  filename = filebase+'.json'
  filename_log = filebase+'.log.json'
  filename_hist = filebase+'.hist.json'
  filename_eval_hist = filebase+'.eval_hist.json'
  filename_real_eval_hist = filebase+'.real_eval_hist.json'
  filename_hist_best = filebase+'.hist_best.json'
  filename_best = filebase+'.best.json'
  
  t = 0

  history = []
  history_best = [] # stores evaluation averages every 25 steps or so
  eval_log = []
  eval_hist = []
  real_hist = []
  best_reward_eval = 0
  best_model_params_eval = None

  max_len = -1 # max time steps (-1 means ignore)
  for generation_i in range(2000): # run for 2k generations
    solutions = es.ask()

    if antithetic:
      seeds = seeder.next_batch(int(es.popsize/2))
      seeds = seeds+seeds
    else:
      seeds = seeder.next_batch(es.popsize)
    packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

    send_packets_to_slaves(packet_list)
    
    response_list_total = receive_packets_from_slaves()
    reward_list_raw = response_list_total[:, :(num_episode)] # get rewards
    time_list_raw = response_list_total[:, (num_episode):]
    if batch_mode == 'min':
        reward_reduced = reward_list_raw.min(axis=1)
    elif batch_mode == 'mean':
        reward_reduced = reward_list_raw.mean(axis=1)
    # actual statistics from non reduced rewards
    mean_time_step = int(np.mean(time_list_raw)*100)/100. 
    max_time_step = int(np.max(time_list_raw)*100)/100.
    avg_reward = int(np.mean(reward_list_raw)*100)/100.
    std_reward = int(np.std(reward_list_raw)*100)/100. 
    es.tell(reward_reduced)

    es_solution = es.result()
    model_params = es_solution[0] # best historical solution
    reward = es_solution[1] # best reward
    curr_reward = es_solution[2] # best of the current batch
    controller.set_model_params(np.array(model_params).round(4))

    r_max = int(np.max(reward_list_raw)*100)/100.
    r_min = int(np.min(reward_list_raw)*100)/100.

    curr_time = int(time.time()) - start_time

    h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

    if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)
    else:
      max_len = -1

    history.append(h)

    with open(filename, 'wt') as out:
      res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

    with open(filename_hist, 'wt') as out:
      res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    sprint(env_name, h)

    if (t == 1):
      best_reward_eval = avg_reward
    if (t % eval_steps == 0): # evaluate on the dream environment with the best agent in the population

      prev_best_reward_eval = best_reward_eval
      model_params_quantized = np.array(es.current_param()).round(4)
      if config_args.dream_env == True:
        reward_eval_list = evaluate_batch(model_params_quantized, max_len=-1, train_mode=0) # evaluate in dream environment
      else:
        reward_eval_list = evaluate_batch(model_params_quantized, max_len=-1, train_mode=-1) # evaluate in real environment

      reward_eval = np.mean(reward_eval_list)
      r_eval_std = np.std(reward_eval_list)
      r_eval_min = np.min(reward_eval_list)
      r_eval_max = np.max(reward_eval_list)
      model_params_quantized = model_params_quantized.tolist()
      improvement = reward_eval - best_reward_eval
      eval_log.append([t, reward_eval, model_params_quantized])
      e_h = (t, reward_eval, r_eval_std, r_eval_min, r_eval_max)
      eval_hist.append(e_h)
      with open(filename_eval_hist, 'wt') as out:
        res = json.dump(eval_hist, out, sort_keys=False, indent=0, separators=(',', ':'))
      with open(filename_log, 'wt') as out:
        res = json.dump(eval_log, out)
      if (len(eval_log) == 1 or reward_eval > best_reward_eval):
        best_reward_eval = reward_eval
        best_model_params_eval = model_params_quantized

        # if new high score in the dream environment test on the real environment
        if config_args.dream_env == True:
          reward_real_eval_list = evaluate_batch(best_model_params_eval, max_len=-1, train_mode=-1) # evaluate in REAL environment
          reward_real_eval = np.mean(reward_real_eval_list)
          r_real_eval_std = np.std(reward_real_eval_list)
          r_real_eval_min = np.min(reward_real_eval_list)
          r_real_eval_max = np.max(reward_real_eval_list)
          real_h = (t, reward_real_eval, r_real_eval_std, r_real_eval_min, r_real_eval_max)
          real_hist.append(real_h)
          with open(filename_real_eval_hist, 'wt') as out:
            res = json.dump(real_hist, out, sort_keys=False, indent=0, separators=(',', ':'))

      else:
        if retrain_mode:
          sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
          es.set_mu(best_model_params_eval)
      with open(filename_best, 'wt') as out:
        res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
      # dump history of best
      curr_time = int(time.time()) - start_time
      best_record = [t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval]
      history_best.append(best_record)
      with open(filename_hist_best, 'wt') as out:
        res = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

      sprint("Eval", t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)

    # increment generation
    t += 1


def main(args):
  global optimizer, num_episode, num_test_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode, env_name, exp_name, batch_mode, config_args

  optimizer = args.controller_optimizer
  num_episode = args.controller_num_episode
  num_test_episode = args.controller_num_test_episode
  eval_steps = args.controller_eval_steps
  num_worker = args.controller_num_worker
  num_worker_trial = args.controller_num_worker_trial
  antithetic = (args.controller_antithetic == 1)
  if antithetic and optimizer != 'oes':
      raise ValueError('OpenES is the only optimizer we support antithetic sampling')
  retrain_mode = (args.controller_retrain == 1)
  cap_time_mode= (args.controller_cap_time == 1)
  seed_start = args.controller_seed_start
  env_name = args.env_name
  exp_name = args.exp_name
  batch_mode = args.controller_batch_mode
  config_args = args

  initialize_settings(args.controller_sigma_init, args.controller_sigma_decay)
  
  sprint("process", rank, "out of total ", comm.Get_size(), "started")
  if (rank == 0):
    master()
  else:
    slave()

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "--allow-run-as-root", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "--allow-run-as-root", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers', nworkers, rank)
    return "child"

if __name__ == "__main__":
  args = PARSER.parse_args()
  if "parent" == mpi_fork(args.controller_num_worker+1): os.exit()
  main(args)
