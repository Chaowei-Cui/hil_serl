import time
import numpy as np
from experiments.mappings import CONFIG_MAPPING

def main():
    exp_name="ram_insertion"
    config=CONFIG_MAPPING[exp_name]()


    env=config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs,info=env.reset()
    hz=10
    try:
        while True:
            action=env.action_space.sample()
            obs,rew,done,trunctaed,info=env.step(action)
            if "intervene_action" in info:
                print("Intervented by Spacemouse")
        time.sleep(1.0/hz)
        if done or trunctaed:
            obs,info=env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__=="__main__":
    main()
