### Description
------------
Implementation of Actor-Critic for OpenAI Gym.

### Requirements
------------
```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage
```
usage: main.py [-h] [--env-name ENV_NAME][--gamma G] [--lr G]
               [--seed N] [--batch_size N] [--num_steps N]
               [--hidden_size N] [--updates_per_step N]
               [--start_steps N] [--replay_size N] [--eval_every N] [--cuda]
```

#### Training

```
python main.py --env-name LunarLander-v2
```

### Arguments
```
PyTorch Actor-Critic Args

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Mujoco Gym environment (default: LunarLander-v2)
  --gamma G             discount factor for reward (default: 0.99)
  --lr G                learning rate (default: 3e-4)
  --seed N              random seed (default: 123456)
  --batch_size N        batch size (default: 256)
  --num_steps N         maximum number of steps (default: 1e6)
  --hidden_size N       hidden size (default: 256)
  --updates_per_step N  model updates per simulator step (default: 1)
  --start_steps N       Steps sampling random actions (default: 1e4)
  --replay_size N       size of replay buffer (default: 1e6)
  --eval_every N        evaluate the gameplay every n steps (default: 50)
  --cuda                run on CUDA (default: False)
```
