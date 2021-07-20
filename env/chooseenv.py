# -*- coding:utf-8  -*-
from env.snakes import SnakeEatBeans
# from env.snake_new import SnakeEatBeans
import configparser
import os


def make(env_type, conf=None):
    config = configparser.ConfigParser() # 初始化configparser
    path = os.path.join(os.path.dirname(__file__), 'config.ini')
    # print(path)
    config.read(path, encoding="utf-8")  # 读取配置文件
    env_list = config.sections()
    conf_dic = {}
    for env_name in env_list:
        conf_dic[env_name] = config[env_name]
    if env_type not in env_list:
        raise Exception("可选环境列表：%s,传入环境为%s" % (str(env_list), env_type))
    if conf:
        conf_dic[env_type] = conf

    env = SnakeEatBeans(conf_dic[env_type])
    return env


if __name__ == "__main__":
    make("sokoban_1p")
