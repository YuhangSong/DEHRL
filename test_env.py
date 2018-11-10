import pybullet_envs
env = pybullet_envs.make('ReacherBulletEnv-v0')
env.reset()
print(env.render())
while True:
    import time
    time.sleep(10)
