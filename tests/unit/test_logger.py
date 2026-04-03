from meta_ai_core.utils.logger import get_logger


def test_get_logger_returns_logger_instance():
    logger = get_logger("meta_ai_core.test")
    assert logger.name == "meta_ai_core.test"
