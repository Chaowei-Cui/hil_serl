import pickle
import cv2
import numpy as np  # Add this import
import time
import numpy as np
from experiments.mappings import CONFIG_MAPPING


def view_picture(data):
    observation = data["observations"]
    image_wrist1 = np.array(observation["wrist_1"])[0]
    image_wrist2 = np.array(observation["wrist_2"])[0]

    # Concatenate images horizontally
    concat_img = np.hstack((image_wrist1, image_wrist2))

    cv2.imshow("wrist1 + wrist2", concat_img)
    cv2.waitKey(0)

def replay_action(env, data):
    _, _ = env.reset()
    actions = data["actions"]
    print(actions)
    next_obs, rew, done, truncated, info = env.step(actions)


pkl_path = "/home/eai/hil-serl/examples/experiments/ram_insertion/classifier_data/ram_insertion_failure_images_2025-09-10_15-13-14.pkl"


if __name__ == "__main__":
    config = CONFIG_MAPPING["ram_insertion"]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        print(data)
        for idx,d in enumerate(data):
            print(f"replay episode {idx}")
            # view_picture(d)
            # replay_action(env, d)

