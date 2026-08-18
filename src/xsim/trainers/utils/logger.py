import os
import sys
import functools
from datetime import datetime

from xsim.utils import Tee


def logged(method):
    """Decorate a ``Trainer`` method so its :class:`TrainLogger` is finalized
    (stdout/stderr restored, log file closed) when the method returns or raises."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self.logger.finalize()
    return wrapper


class TrainLogger:
    """Mirrors everything printed to stdout/stderr into ``<experiment_dir>/train.log``.

    Wraps the *current* streams (not ``sys.__stdout__``) so notebook display
    keeps working; the file captures the same bytes, including the strategy's
    verbose densify/prune counts. Rich progress bars are written to the real
    stderr (the tee is unwrapped in ``xsim.utils.progress``), so their live
    redraws stay out of the log.
    """

    def __init__(self, experiment_dir: str, config_path: str, enabled: bool = True):
        self.experiment_dir = experiment_dir
        self.config_path = config_path
        # When disabled (e.g. a trainer reused from a pretrained experiment via
        # ``from_experiment``) ``start`` is a no-op, so the original
        # ``train.log`` is left untouched.
        self.enabled = enabled
        self.log_file = None
        self._orig_stdout = None
        self._orig_stderr = None

    def start(self):
        if not self.enabled or self.log_file is not None:
            return

        log_path = os.path.join(self.experiment_dir, 'train.log')
        self.log_file = open(log_path, 'a', buffering=1)  # line-buffered
        self.log_file.write('\n===== run started {} (config: {}) =====\n'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.config_path))

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, self.log_file)
        sys.stderr = Tee(sys.stderr, self.log_file)

        print('Logging to {}'.format(log_path))

    def finalize(self):
        if self.log_file is None:
            return

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.log_file.close()
        self.log_file = None
