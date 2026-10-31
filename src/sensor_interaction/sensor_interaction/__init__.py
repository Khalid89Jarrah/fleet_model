from gymnasium.envs.registration import register
from sensor_interaction.fleet_model_world import FleetModel
from sensor_interaction.fast_drone_node import FastDroneNode as _NodeClass


def create_fleet_model_env(**kwargs):
    main_node = _NodeClass()
    return FleetModel(main_node=main_node)


register(
    id="FleetModel-v0",
    entry_point="sensor_interaction:create_fleet_model_env",
    max_episode_steps=400,
)
