from minecraft import MineCraft, minecraft_global_setup

def main():

    env = MineCraft(obs_size=1000, obs_type='rgb')
    env.set_render(True)

    minecraft_global_setup()

    obs = env.reset()
    action = env.key_map_to_action[cv2.waitKey(0)]

    while True:

        if action==ord('j'):
            cv2.destroyAllWindows()
            break

        obs, reward, done, info = env.step(action)
        # print(obs.shape)
        if done:
            env.reset()
        action = env.key_map_to_action[cv2.waitKey(0)]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
