import time

class Utils:
    def timeit(func):
        def inner(*args,**kwargs):
            start = time.time()
            res = func(*args,**kwargs)
            end = time.time()
            print(f"Elapsed Time: {(end-start):.4f}s")
            return res
        return inner