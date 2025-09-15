import time
from src.handsfree_reels_scroller.gesture_recognition import detect_swipe, SwipeConfig
from src.handsfree_reels_scroller.actions import Action


def make_points(start_t: float, coords):
    return [(start_t + dt, x, y) for dt, x, y in coords]


def test_vertical_next_swipe():
    t0 = time.time()
    points = make_points(t0, [
        (0.0, 0.5, 0.2),
        (0.2, 0.5, 0.45),  # moved downward (y increases)
    ])
    cfg = SwipeConfig(min_displacement=0.15, axis="vertical", max_duration=0.5)
    assert detect_swipe(points, cfg) == Action.NEXT


def test_vertical_prev_swipe():
    t0 = time.time()
    points = make_points(t0, [
        (0.0, 0.5, 0.6),
        (0.2, 0.5, 0.3),  # upward
    ])
    cfg = SwipeConfig(min_displacement=0.15, axis="vertical", max_duration=0.5)
    assert detect_swipe(points, cfg) == Action.PREV


def test_displacement_too_small():
    t0 = time.time()
    points = make_points(t0, [
        (0.0, 0.5, 0.50),
        (0.1, 0.5, 0.57),  # only 0.07 movement
    ])
    cfg = SwipeConfig(min_displacement=0.15, axis="vertical", max_duration=0.5)
    assert detect_swipe(points, cfg) is None


def test_duration_too_long():
    t0 = time.time()
    points = make_points(t0, [
        (0.0, 0.5, 0.2),
        (1.0, 0.5, 0.5),  # 1.0s > max_duration
    ])
    cfg = SwipeConfig(min_displacement=0.15, axis="vertical", max_duration=0.5)
    assert detect_swipe(points, cfg) is None


def test_horizontal_swipe_next():
    t0 = time.time()
    points = make_points(t0, [
        (0.0, 0.2, 0.5),
        (0.2, 0.5, 0.5),  # rightwards
    ])
    cfg = SwipeConfig(min_displacement=0.15, axis="horizontal", max_duration=0.5)
    assert detect_swipe(points, cfg) == Action.NEXT
