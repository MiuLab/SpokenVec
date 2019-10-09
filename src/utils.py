import time
import re


def get_time():
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    return Y, M, D, h, m, s


def print_time_info(string):
    Y, M, D, h, m, s = get_time()
    _string = re.sub('[ \n]+', ' ', string)
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(
        Y, M, D, h, m, s, _string))
