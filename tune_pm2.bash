#!/bin/bash

echo "sac pm2 tune"
wandb agent multi-task-rl/concurrent_composition/shzwzw7t --count 25

echo "msf pm2 tune"
wandb agent multi-task-rl/concurrent_composition/wqrsy4o8 --count 25

echo "sfgpi pm2 tune"
wandb agent multi-task-rl/concurrent_composition/gz92w98g --count 25

echo "dac pm2 tune"
wandb agent multi-task-rl/concurrent_composition/qn3mb46x --count 25

echo "dacgpi pm2 tune"
wandb agent multi-task-rl/concurrent_composition/l68v9r62 --count 25


