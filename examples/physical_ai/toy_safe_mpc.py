"""Educational 1D receding-horizon planner; SIMULATION ONLY, not robot control.

Standard-library only. The predictive model and simulated plant have slightly
mis-matched drag coefficients. A separate safety shield checks proposed actions.
"""
from dataclasses import dataclass
from itertools import product

DT = 0.2
ACTIONS = (-1.0, 0.0, 1.0)
MAX_V = 0.9
MIN_X, MAX_X = -0.2, 1.50   # Simple hard workspace limits in metres
GOAL = 1.10


@dataclass(frozen=True)
class State:
    x: float  # position in metres
    v: float  # velocity in metres/second


def transition(state: State, acceleration: float, drag: float) -> State:
    """Toy deterministic dynamics; deliberately not a physical robot model."""
    v_next = max(-MAX_V, min(MAX_V, drag * state.v + DT * acceleration))
    x_next = state.x + DT * v_next
    return State(x_next, v_next)


def is_safe(state: State) -> bool:
    return MIN_X <= state.x <= MAX_X and abs(state.v) <= MAX_V


def candidate_plan(state: State, horizon: int = 5) -> float:
    """Enumerate candidate acceleration sequences and return first action."""
    best = (float("inf"), 0.0)
    for actions in product(ACTIONS, repeat=horizon):
        predicted = state
        effort, speed = 0.0, 0.0
        feasible = True
        for action in actions:
            predicted = transition(predicted, action, drag=0.985)
            if not is_safe(predicted):
                feasible = False
                break
            effort += action * action
            speed += predicted.v * predicted.v
        if not feasible:
            continue
        cost = 15.0 * (predicted.x - GOAL)**2 + 0.08 * effort + 0.04 * speed
        if cost < best[0]:
            best = (cost, actions[0])
    return best[1]


def safety_shield(observed: State, proposed_action: float) -> float:
    """Independent one-step admissibility check; refuse unsafe commands."""
    if proposed_action not in ACTIONS:
        raise ValueError("Command outside allowed discrete action set")
    # Conservative dynamics for this toy: check several plausible drag values.
    for drag in (0.93, 0.985, 1.0):
        if not is_safe(transition(observed, proposed_action, drag)):
            return 0.0
    return proposed_action


def run(steps: int = 40) -> list[State]:
    """Observe -> plan -> shield -> execute -> observe again."""
    measured = State(0.0, 0.0)
    trace = [measured]
    for _ in range(steps):
        proposed = candidate_plan(measured)
        approved = safety_shield(measured, proposed)
        measured = transition(measured, approved, drag=0.94)  # "Real" toy plant
        if not is_safe(measured):
            raise AssertionError("Safety limit violated in simulation")
        trace.append(measured)
    return trace


if __name__ == "__main__":
    trace = run()
    print(f"Final toy position: {trace[-1].x:.3f} m")
    print(f"Final toy velocity: {trace[-1].v:.3f} m/s")
    print(f"Distance to goal: {abs(trace[-1].x - GOAL):.3f} m")
    print("Safety satisfied at all steps:", all(is_safe(s) for s in trace))