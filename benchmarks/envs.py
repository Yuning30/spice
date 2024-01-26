def get_env_from_name(name):

    if name == 'acc':
        from .acc import AccEnv
        return AccEnv()
    if name == 'car_racing':
        from .car_racing import CarRacingEnv
        return CarRacingEnv()
    if name == 'mid_obstacle':
        from .mid_obstacle import MidObstacleEnv
        return MidObstacleEnv()
    if name == 'mountain_car':
        from .mountain_car import MountainCarEnv
        return MountainCarEnv()
    if name == 'noisy_road':
        from .noisy_road import NoisyRoadEnv
        return NoisyRoadEnv()
    if name == 'noisy_road_2d':
        from .noisy_road_2d import NoisyRoad2dEnv
        return NoisyRoad2dEnv()
    if name == 'obstacle':
        from .obstacle import ObstacleEnv
        return ObstacleEnv()
    if name == 'pendulum':
        from .pendulum import PendulumEnv
        return PendulumEnv()
    if name == 'road':
        from .road import RoadEnv
        return RoadEnv()
    if name == 'road_2d':
        from .road_2d import Road2dEnv
        return Road2dEnv()
    if name == "cartpole":
        from .cartpole import CartPoleEnv
        return CartPoleEnv()
    if name == "cartpole_move":
        from .cartpole_move import CartPoleMoveEnv
        return CartPoleMoveEnv()
    if name == "cartpole_swing":
        from .cartpole_swing import CartPoleSwingEnv
        return CartPoleSwingEnv()
    if name == "lalo":
        from .lalo import LaloeEnv
        return LaloeEnv()
    else:
        raise RuntimeError("Unkonwn environment: " + name)
