from pandas_profiling import config


def dask_diagnose(func,):
    def register_profilers():
        profilers = {}
        if config["dask"]["progress_bar"].get(bool):
            from dask.diagnostics import ProgressBar

            profilers["pbar"] = ProgressBar()

        if config["dask"]["profiler"].get(bool):
            from dask.diagnostics import Profiler

            profilers["prof"] = Profiler()

        if config["dask"]["resource_profiler"].get(bool):
            from dask.diagnostics import ResourceProfiler

            profilers["rprof"] = ResourceProfiler(dt=0.25)

        if config["dask"]["cache_profiler"].get(bool):
            from dask.diagnostics import CacheProfiler

            profilers["cprof"] = CacheProfiler()

        # register the profilers
        for name, dia in profilers.items():
            dia.register()
        return profilers

    def unregister_profilers(profilers):
        # unregister the profilers
        for name, dia in profilers.items():
            dia.unregister()

        # visualize the profilers immediately
        if set(profilers.keys()) != {"pbar"}:
            from dask.diagnostics import visualize

            visualize([prof for name, prof in profilers.items() if name != "pbar"])

    def wrapper(*args, **kw):
        profilers = register_profilers()
        res = func(*args, **kw)
        unregister_profilers(profilers)
        return res

    return wrapper
