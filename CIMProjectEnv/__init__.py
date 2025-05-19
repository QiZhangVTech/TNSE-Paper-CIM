from gym.envs.registration import register

register(
    id = 'InfluenceSpreadEnv-v1',
    entry_point = 'CIMProjectEnv.envs:InfluenceSpreadEnv_v1',
)

register(
    id = 'InfluenceSpreadEnv-v2',
    entry_point = 'CIMProjectEnv.envs:InfluenceSpreadEnv_v2',
)