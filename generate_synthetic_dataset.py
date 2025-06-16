import os, sys
from captcha.image import ImageCaptcha
import random, string, argparse, tqdm, yaml

def generate(cfg, n_samples: int):
    CHARS = cfg["char_set"]
    gen   = ImageCaptcha(width=cfg["img_width"], height=cfg["img_height"])
    os.makedirs(cfg["data_dir"], exist_ok=True)

    for i in tqdm.tqdm(range(n_samples)):
        text = "".join(random.choices(CHARS, k=random.randint(4, 8)))
        gen.write(text, os.path.join(cfg["data_dir"], f"{text}_{i}.png"))

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=20000)
    args = parser.parse_args()
    generate(cfg, args.num)
