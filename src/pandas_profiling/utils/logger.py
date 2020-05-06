import functools
from pathlib import Path
from time import time, strftime, gmtime

import pandas as pd


class Logger:
    logs = {}
    start_time = time()
    running_functions = set()
    mutiprocess_functions = {
        "describe_1d"
    }  # to make mutiprocess_functions level stable

    def __init__(self):
        raise AttributeError("Loger can not be instantiated")

    @staticmethod
    def get_wrapper(func_or_description):
        """Use as a decorator

        Note:
            Can either be used like @Loger.log or @Loger.log("Test Description Name")
        """

        if hasattr(
            func_or_description, "__call__"
        ):  # it is function, use decorator like @Loger.log

            def wrapper(*args, **kwargs):

                description = Logger._rename_description(
                    func_or_description.__qualname__
                )
                Logger._log_start(description)
                res = func_or_description(*args, **kwargs)
                Logger._log_finished(description)

                return res

            return wrapper
        elif isinstance(
            func_or_description, str
        ):  # it is description, use decorator like @Loger.log("Test function")

            def decorator(func, description):
                def wrapper(*args, **kwargs):
                    new_description = Logger._rename_description(description)
                    Logger._log_start(new_description)
                    res = func(*args, **kwargs)
                    Logger._log_finished(new_description)

                    return res

                return wrapper

            decorator = functools.partial(decorator, description=func_or_description)
            return decorator

    log = get_wrapper  # alias

    @classmethod
    def to_str(cls, time_fliter=0, proportion=False):
        """Render saved records to string

        Args:
            time_fliter: Function run times less than this value will not be rendered
            proportion: Display the proportion of function run times
        Returns:
            This function returns the function calls information, containing:
                - start_time: The time when function started
                - end_time: The time when function end
                - (optional) proportion: The percentage of time function takes up
                - level: The hierarchy of function calls
                - description: The name of function
                - use_time : The time function takes ip
        """

        res = ""
        if proportion:
            total_time = cls.total_time()
            for k, v in cls.logs.items():
                if v[2] > time_fliter:
                    res += (
                        "[{start_time}~{end_time} {proportion:2d}%] {level} "
                        "{description} spend {use_time:.2f} s\n".format(
                            start_time=Logger._str_time(v[0]),
                            end_time=Logger._str_time(v[1]),
                            proportion=int((100 * v[2]) // total_time),
                            level="├" + "-" * v[3],
                            description=k,
                            use_time=v[2],
                        )
                    )
        else:
            for k, v in cls.logs.items():
                if v[2] > time_fliter:
                    res += (
                        "[{start_time}~{end_time}] {level} "
                        "{description} spend {use_time:.2f} s\n".format(
                            start_time=Logger._str_time(v[0]),
                            end_time=Logger._str_time(v[1]),
                            level="-" * v[3],
                            description=k,
                            use_time=v[2],
                        )
                    )

        return res

    @classmethod
    def to_df(cls):
        return pd.DataFrame(cls.logs)

    @classmethod
    def save(cls, output_file, **kwargs):
        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))
        with output_file.open("w", encoding="utf8") as f:
            f.write(cls.to_str(**kwargs))

    @classmethod
    def clear(cls):
        """Clear current record"""
        cls.logs = {}
        cls.start_time = time()
        cls.running_functions = set()

    @classmethod
    def total_time(cls):
        return time() - cls.start_time

    @classmethod
    def add_log(cls, description, start_time=None, end_time=None):
        if start_time is not None:
            cls.logs[description][0] = start_time
            if not description.endswith(">"):
                cls.running_functions.add(description)
            cls.logs[description][3] = len(cls.running_functions)  # save level
        if end_time is not None:
            cls.logs[description][1] = end_time
            cls.logs[description][2] = (
                cls.logs[description][1] - cls.logs[description][0]
            )
            if not description.endswith(">"):
                cls.running_functions.remove(description)

    @staticmethod
    def _str_time(time_stamp):
        return strftime("%H:%M:%S", gmtime(time_stamp))

    @staticmethod
    def _log_start(description):
        Logger.add_log(description, start_time=time())

    @staticmethod
    def _log_finished(description):
        Logger.add_log(description, end_time=time())

    @staticmethod
    def _rename_description(description):
        """Prevents function name duplication"""
        new_description = description
        i = 1
        while new_description in Logger.logs.keys():
            new_description = description + f"<{i}>"
            i += 1
        Logger.logs[new_description] = [None] * 4  # start_time,end_time,use_time,level
        return new_description
