import json
import os
import yaml

import sys

sys.path.append(os.path.join(os.getcwd(), 'config.yml'))
config_file = os.path.join(r'/home/ps/development_sources/Data_PipeLine/mysql_toolkits/config/config.yml')
with open(config_file, 'r', encoding='utf8') as temp:
    config_lib = yaml.load(temp, yaml.BaseLoader)

vendor = config_lib.get('vendor', {})
sql_con = config_lib.get('sql', {})
no_sql_config = config_lib.get('no_sql', {})  # sql登录设置
global_configs = config_lib.get('global', {})


def get_file_tree(folder, file_name, is_contain=False):
    for root, _, files in os.walk(folder):
        if file_name in files:
            yield os.path.join(root, file_name)
        if is_contain:
            for file in files:
                if file_name in file:
                    yield os.path.join(root, file)


def config_embedding(initial_val=None, env_val=None, file_val=None, default=None, config_type=0):
    if initial_val is not None:
        return initial_val

    config_list = [env_val, file_val, default]

    config_val = config_list.pop(config_type)

    if config_val is not None:
        return config_val

    while len(config_list) != 0:
        config_val = config_list.pop(0)
        if config_val is not None:
            return config_val

    return


if __name__=='__main__':
    print(sql_con)