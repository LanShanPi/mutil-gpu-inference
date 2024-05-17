import redis
redis_config = {
        'host': '127.0.0.1',
        'port': 6379,
        'db': 0,
     }
r = redis.StrictRedis(**redis_config)
a = r.hgetall("id")
b = r.hkeys("id")
c = r.hget("id",b[0].decode("utf-8"))
print(c.decode("utf-8"))

