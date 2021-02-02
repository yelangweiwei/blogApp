import redis
from rediscluster import RedisCluster

def conn_redis():
    r = redis.StrictRedis(host='10.10.9.159',port=6379,password=123456,db=0,decode_responses=True)
    return r
def conn_pool_redis():
    pool =redis.ConnectionPool(host='10.10.9.159',port=6379,db=0,password=123456,decode_responses=True)
    r = redis.Redis(connection_pool=pool)

    r.zadd('runn1',{1:1})
    r.rename('runn1','runn2')
    print(r.zrange('runn1',0,-1))
    print(r.zrange('runn2',0,-1))
    print(r.randomkey())
    print(type('runn1'))
    print(r.dbsize())


def conn_redis_cluster():
    start_up_nodes = [
        {'host':'192.168.22.69','port':'7000'},
        {'host':'192.168.22.69','port':'7001'},
        {'host':'192.168.22.69','port':'7003'}
    ]
    conn = RedisCluster(startup_nodes=start_up_nodes,decode_responses=True)
    print(conn.set('runn','1'))
    print(conn.get('runn'))


if __name__ == '__main__':
    try:
        conn_redis_cluster()

    except Exception as e:
        print(e)