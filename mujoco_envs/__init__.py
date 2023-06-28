from gym.envs.registration import register

register(id="MJCartpole-v0", entry_point="mujoco_envs.base_envs.cartpole:CartpoleEnv", max_episode_steps=200)

register(id="MJPusher-v0", entry_point="mujoco_envs.base_envs.pusher:PusherEnv", max_episode_steps=150)

register(id="MJReacher-v0", entry_point="mujoco_envs.base_envs.reacher:Reacher3DEnv", max_episode_steps=150)

register(id="MJHalfCheetah-v0", entry_point="mujoco_envs.base_envs.half_cheetah:HalfCheetahEnv", max_episode_steps=1000)

###############################
#          anom envs          #
###############################

register(id="AnomMJCartpole-v0", entry_point="mujoco_envs.mod_envs:AnomCartpoleEnv", max_episode_steps=200)

register(id="AnomMJHalfCheetah-v0", entry_point="mujoco_envs.mod_envs:AnomHalfCheetahEnv", max_episode_steps=1000)

register(id="AnomMJPusher-v0", entry_point="mujoco_envs.mod_envs:AnomPusherEnv", max_episode_steps=150)

register(id="AnomMJReacher-v0", entry_point="mujoco_envs.mod_envs:AnomReacher3DEnv", max_episode_steps=150)

###############################
#           mod envs          #
###############################

register(
    id="ModMJCartpole-v0",
    entry_point="mujoco_envs.mod_envs:ModCartpoleEnv",
    max_episode_steps=200,
)

register(
    id="ModMJHalfCheetah-v0",
    entry_point="mujoco_envs.mod_envs:ModHalfCheetahEnv",
    max_episode_steps=1000,
)

register(id="ModMJPusher-v0", entry_point="mujoco_envs.mod_envs:ModPusherEnv", max_episode_steps=150)

register(id="ModMJReacher-v0", entry_point="mujoco_envs.mod_envs:ModReacher3DEnv", max_episode_steps=150)
