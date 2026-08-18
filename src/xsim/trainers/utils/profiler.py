import os

from xsim.utils import profiling


class TrainProfiler:
    def __init__(self, cfg, experiment_dir: str):
        self.experiment_dir = experiment_dir
        p = cfg.get('profile', None)
        self.active = bool(p)
        self.profiler = None
        self.window = 0
        if self.active:
            self.profiler = self.build(p)

    def build(self, p):
        from torch.profiler import (
            profile, schedule, ProfilerActivity, tensorboard_trace_handler
        )
        self.window = p.get('wait', 500) + p.get('warmup', 5) + p.get('active', 10)
        trace_dir = os.path.join(self.experiment_dir, 'trace')
        os.makedirs(trace_dir, exist_ok=True)
        print('profiling: capturing trace to', trace_dir)
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=p.get('wait', 500), warmup=p.get('warmup', 5),
                active=p.get('active', 10), repeat=1),
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=p.get('record_shapes', False),
            with_stack=p.get('with_stack', True),
            profile_memory=p.get('profile_memory', False),
        )

    def __enter__(self):
        if self.active:
            self.profiler.start()
            profiling.set_enabled(True)

        return self

    def __exit__(self, *exc):
        if self.active:
            profiling.set_enabled(False)
            self.profiler.stop()

        return False

    def step(self, step: int) -> bool:
        if not self.active:
            return False

        self.profiler.step()

        if step >= self.window:
            print('profiling window complete; stopping at step', step)
            return True

        return False
