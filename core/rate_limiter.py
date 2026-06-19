from collections import defaultdict, deque
import time


class RateLimiter:
    """Sliding-window rate limiter keyed by user identifier."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self._windows: dict[str, deque] = defaultdict(deque)
        self._max = max_requests
        self._window = window_seconds

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        window = self._windows[user_id]
        while window and window[0] < now - self._window:
            window.popleft()
        if len(window) >= self._max:
            return False
        window.append(now)
        return True
