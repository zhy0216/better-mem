from fastapi import Depends

from src.store.database import get_pool
from src.store.cache import get_redis


def get_db():
    return get_pool()


def get_cache():
    return get_redis()
