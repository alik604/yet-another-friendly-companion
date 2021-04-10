from models.engine import load_engine
from models.d4pg.engine import Engine as D4PGEngine

from utils.utils import read_config

if __name__ == "__main__":
    #CONFIG_PATH = "configs/LunarLanderContinuous_d4pg.yml"
    CONFIG_PATH = "d4pg/configs/follow_rl.yml" # d4pg-pytorch/
    config = read_config(CONFIG_PATH)

    engine = D4PGEngine(config) #load_engine(config)
    engine.train()
