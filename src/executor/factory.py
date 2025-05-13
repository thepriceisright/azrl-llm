from utils import get_logger, get_config

def get_executor():
    """
    Factory function that returns either a Docker-based executor or a mock executor
    based on the configuration.
    
    Returns:
        Either a Docker-based or mock executor instance
    """
    config = get_config()
    logger = get_logger("executor_factory")
    
    # Check if we should use the mock executor
    use_mock = config.get("executor.use_mock", False)
    
    if use_mock:
        logger.log("Using mock executor (no Docker)")
        from src.executor.mock_executor import get_mock_executor
        return get_mock_executor()
    else:
        logger.log("Using Docker-based executor")
        from src.executor.executor_client import get_executor as get_docker_executor
        return get_docker_executor() 