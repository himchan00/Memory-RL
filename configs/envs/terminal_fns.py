def finite_horizon_terminal(done: bool, info: dict) -> bool:
    return done


def infinite_horizon_terminal(done: bool, info: dict) -> bool:
    if not done or "TimeLimit.truncated" in info:
        terminal = False
    else:
        terminal = True
    return terminal
