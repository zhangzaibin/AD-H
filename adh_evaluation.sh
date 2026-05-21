#!/bin/bash
export CARLA_ROOT=./CARLA_0.9.15
export CARLA_PORT=$(($RANDOM % 1000 + 16000))
export CARLA_HOST=localhost
export CARLA_DEVICE=0
export LEADERBOARD_ROOT=./leaderboard
export SCENARIO_RUNNER_ROOT=./scenario_runner
export TEAM_CODE_ROOT=./team_code                           # path to the team code
export TEAM_AGENT=${TEAM_CODE_ROOT}/adh_agent.py            # agent
export TEAM_CONFIG=${TEAM_CODE_ROOT}/adh_agent_config.py    # model checkpoint, not required for expert

export PYTHONPATH=${PYTHONPATH}:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=${PYTHONPATH}:${LEADERBOARD_ROOT}
export PYTHONPATH=${PYTHONPATH}:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=${PYTHONPATH}:${TEAM_CODE_ROOT}
export PYTHONPATH=${PYTHONPATH}:./quick_imports

export ROUTES=${LEADERBOARD_ROOT}/data/langauto/benchmark_tiny.xml
export SCENARIOS=${LEADERBOARD_ROOT}/data/official/all_towns_traffic_scenarios_public.json
export SAVE_PATH=data/eval                                  # path for saving episodes while evaluating
export CHECKPOINT_ENDPOINT=${SAVE_PATH}/sample_result.json  # results file
export CHALLENGE_TRACK_CODENAME=SENSORS
export TRAFFIC_MANAGER_PORT=$(($CARLA_PORT+500))            # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1                                        # multiple evaluation runs
export RESUME=False

bash ${CARLA_ROOT}/CarlaUE4.sh --world-port=$CARLA_PORT --host=$CARLA_HOST -graphicsadapter=$CARLA_DEVICE &
sleep 10
bash leaderboard/scripts/run_evaluation.sh
