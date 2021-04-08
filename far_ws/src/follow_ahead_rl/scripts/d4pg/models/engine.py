from .d4pg.engine import Engine as D4PGEngine


def load_engine(config):
    return D4PGEngine(config)
