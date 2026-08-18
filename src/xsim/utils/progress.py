import sys
import time
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from .logging import Tee


class RateColumn(ProgressColumn):
    """Iterations per second, from the task's tracked speed."""

    def render(self, task) -> Text:
        speed = task.speed
        if not speed:
            return Text('?  it/s', style='progress.data.speed')
        return Text('{:.2f} it/s'.format(speed), style='progress.data.speed')

_console: Console | None = None

# Keys of the warnings warn_once has already emitted.
_warned: set[str] = set()
_progress: Progress | None = None  # the single live Progress, when one is active

# How often (in completed iterations) to write a plain speed/ETA snapshot of a
# live bar into ``train.log``. The live ANSI bar itself stays out of the log, so
# these periodic lines are the only progress trace it leaves there.
LOG_EVERY = 1000


def _real_stderr():
    stream = sys.stderr
    while isinstance(stream, Tee):
        stream = stream.stream
    return stream


def _log_file():
    """The file the trainer's :class:`Tee` mirrors output into, or ``None`` when
    no logging Tee is installed (e.g. plain notebook use)."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, Tee):
            return stream.file
    return None


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def console() -> Console:
    """Shared console; auto-renders in Jupyter, else writes to the real stderr."""
    global _console
    if _console is None:
        _console = Console() if _in_notebook() else Console(file=_real_stderr())
    return _console


def _columns():
    return (
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        RateColumn(),
        TextColumn('{task.fields[postfix]}'),
    )


@contextmanager
def _live():
    """Yield the shared Progress, starting/stopping it around the outermost use
    so nested/sequential bars all share one live display."""
    global _progress
    created = False
    if _progress is None:
        # Don't let Rich's Live hijack stdout/stderr: its redirect re-emits any
        # captured ``print`` through our console (which targets the *un-Tee'd*
        # real stderr), so those lines would bypass ``train.log``. With the
        # redirect off, plain ``print``s go straight to the Tee'd stdout and are
        # logged as intended, while only the live ANSI redraws stay out.
        _progress = Progress(
            *_columns(), console=console(),
            redirect_stdout=False, redirect_stderr=False)
        _progress.start()
        created = True
    try:
        yield _progress
    finally:
        if created:
            _progress.stop()
            _progress = None


def _fmt(v):
    return '{:.4g}'.format(v) if isinstance(v, float) else str(v)


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return '?'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return '{:d}:{:02d}:{:02d}'.format(h, m, s)


def _log_snapshot(task, description: str):
    """Append a plain ``speed`` / ``ETA`` line for ``task`` to ``train.log``.

    Written straight to the log file (never the terminal), so it doesn't disturb
    the live bar, which already shows the same figures on screen.
    """
    f = _log_file()
    if f is None:
        return
    total = '{:d}'.format(int(task.total)) if task.total is not None else '?'
    f.write('[{}] {}/{}  {:.2f} it/s  ETA {}\n'.format(
        description, int(task.completed), total,
        task.speed or 0.0, _fmt_time(task.time_remaining)))
    f.flush()


class Bar:
    """Handle over a Rich task with a tqdm-ish ``update`` / ``set_postfix`` API."""

    def __init__(self, progress: Progress, task_id, description: str = 'working',
                 log_every: int = LOG_EVERY):
        self._p = progress
        self._t = task_id
        self._description = description
        self._log_every = log_every
        self._milestone = 0

    def _maybe_log(self):
        task = self._p._tasks[self._t]
        milestone = int(task.completed) // self._log_every if self._log_every else 0
        if self._log_every and milestone > self._milestone:
            self._milestone = milestone
            _log_snapshot(task, self._description)

    def update(self, advance: int = 1):
        self._p.advance(self._t, advance)
        self._maybe_log()

    def set_postfix(self, **kwargs):
        self._p.update(self._t, postfix=', '.join(
            '{}={}'.format(k, _fmt(v)) for k, v in kwargs.items()))

    def set_description(self, description: str):
        self._p.update(self._t, description=description)


class _NullBar:
    def update(self, advance: int = 1): pass
    def set_postfix(self, **kwargs): pass
    def set_description(self, description: str): pass


def _infer_total(iterable, total):
    if total is not None:
        return total
    try:
        return len(iterable)
    except TypeError:
        return None


def track(iterable, description: str = 'working', total: int | None = None,
          log_every: int = LOG_EVERY):
    """Iterate ``iterable`` while advancing a shared progress bar.

    Every ``log_every`` iterations a plain speed/ETA snapshot is written to
    ``train.log`` (pass ``log_every=0`` to disable)."""
    total = _infer_total(iterable, total)
    with _live() as p:
        task = p.add_task(description, total=total, postfix='')
        bar = Bar(p, task, description, log_every)
        for item in iterable:
            yield item
            bar.update()


@contextmanager
def progress_bar(description: str | None = 'working', total: int | None = None,
                 log_every: int = LOG_EVERY):
    """Manual bar. ``description=None`` disables it (yields a no-op handle).

    Every ``log_every`` calls to ``update`` a plain speed/ETA snapshot is
    written to ``train.log`` (pass ``log_every=0`` to disable)."""
    if description is None:
        yield _NullBar()
        return
    with _live() as p:
        task = p.add_task(description, total=total, postfix='')
        yield Bar(p, task, description, log_every)


def warn_once(message: str, key: str = None):
    """Print a warning to the shared console the first time it is raised.

    ``key`` identifies the warning for deduplication; it defaults to the
    message itself. Pass one when the text carries varying detail (a count, a
    tensor shape) that would otherwise turn one warning into many. Callers on
    the training path rely on this: a warning about how the model is composed
    says nothing new on iteration two.
    """
    key = message if key is None else key
    if key in _warned:
        return

    _warned.add(key)
    console().print('[bold yellow]![/] {}'.format(message))


@contextmanager
def stage(name: str):
    """Announce a high-level phase; times it and prints a completion line."""
    c = console()
    c.print('[bold blue]▶[/] [bold]{}[/]'.format(name))
    start = time.perf_counter()
    try:
        yield
    finally:
        c.print('  [green]✓[/] {} [dim]({:.1f}s)[/]'.format(
            name, time.perf_counter() - start))
