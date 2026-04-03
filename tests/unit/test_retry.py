from meta_ai_core.utils.retry import retry_with_backoff


def test_retry_eventually_succeeds():
    state = {"n": 0}

    @retry_with_backoff(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)
    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert state["n"] == 3
