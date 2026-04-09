"""
pytest 配置文件
"""
import pytest

# 配置 anyio 只使用 asyncio 后端
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "anyio: mark test as an async test using anyio"
    )


# 限制 anyio 只使用 asyncio 后端
@pytest.fixture
def anyio_backend():
    return "asyncio"