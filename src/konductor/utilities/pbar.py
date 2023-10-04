import time
import os
import itertools
from threading import Thread, Event, Lock
from datetime import timedelta
from typing import Any, List
import enum

from tqdm.auto import tqdm
from colorama import Fore


class PbarType(enum.Enum):
    TQDM = enum.auto()
    LIVE = enum.auto()
    TIMED = enum.auto()
    INTERVAL = enum.auto()


def pbar_wrapper(func, pbar_type: PbarType = PbarType.LIVE, *pbar_args, **pbar_kwargs):
    """Function must take pbar as a key-word argument"""
    pbar_t = {
        PbarType.TQDM: tqdm,
        PbarType.LIVE: LivePbar,
        PbarType.TIMED: TimedPbar,
        PbarType.INTERVAL: IntervalPbar,
    }[pbar_type]

    def with_pbar(*args, **kwargs):
        with pbar_t(*pbar_args, **pbar_kwargs) as pbar:
            func(*args, **kwargs, pbar=pbar)

    return with_pbar


class BasePbar:
    """Use context manager if possible to ensure that thread is cleaned up
    even if the total is not reached."""

    # The amount of the terminal string that's just formatting
    # This increases the "string length" but not its display length
    # So it has to be compensated for
    DEFAULT_N_COLS = 120

    _fmt_len = len(
        f"{Fore.GREEN}{Fore.YELLOW}{Fore.RED}{Fore.RESET}"
        f"{Fore.YELLOW}{Fore.RESET}{Fore.YELLOW}{Fore.RESET}"
        f"{Fore.GREEN}{Fore.YELLOW}{Fore.RED}{Fore.RESET}"
    )

    def __init__(
        self, total: int, desc: str = "Running", ncols: int | None = None
    ) -> None:
        self.total = total
        self.ncols = ncols
        self._desc = desc
        self.n = 0
        self.s_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def elapsed(self) -> timedelta:
        """Elapsed time"""
        return timedelta(seconds=time.time() - self.s_time)

    def elapsed_str(self) -> str:
        """Elapsed time string with microseconds removed"""
        return str(self.elapsed()).split(".")[0]

    def estimated(self) -> timedelta:
        """Estimated completion time"""
        if self.n == 0:
            return timedelta(hours=999)
        time_per_iter = self.elapsed() / self.n
        return time_per_iter * (self.total - self.n)

    def estimated_str(self) -> str:
        """Estimated completion time string with microseconds removed"""
        return str(self.estimated()).split(".")[0]

    def make_start(self):
        n_digits = len(str(self.total))
        start_str = (
            f"{Fore.BLUE}{self._desc}: {Fore.GREEN}{self.n:0{n_digits}}"
            f"{Fore.YELLOW}/{Fore.RED}{self.total}{Fore.RESET}"
        )
        return start_str

    def make_end(self):
        end_str = (
            f"Elapsed: {Fore.YELLOW}{self.elapsed_str()}{Fore.RESET} "
            f"Est: {Fore.YELLOW}{self.estimated_str()}{Fore.RESET}"
        )
        return end_str

    def _get_cols(self):
        """Get number of columns for pbar"""
        if self.ncols is not None:
            return self.ncols
        try:
            return os.get_terminal_size().columns
        except OSError:
            return BasePbar.DEFAULT_N_COLS

    def make_pbar(self, prog_char: str, start_end_len: int):
        """Progress character"""
        ncols = self._get_cols()
        print("\r" + " " * (ncols - 2), end="\r")  # Clear the terminal
        ncols -= start_end_len - self._fmt_len // 2
        done_bars = ncols * self.n // self.total

        bar_str = (
            f"{Fore.GREEN}{'█'*done_bars}{Fore.YELLOW}{prog_char}"
            f"{Fore.RED}{'-'*(ncols - done_bars)}{Fore.RESET}"
        )
        return bar_str

    def set_description(self, desc: str):
        self._desc = desc

    def update(self, update):
        self.n += update


class ThreaddedPbar(BasePbar, Thread):
    """
    Runs pbar as a thread which updates at a specified frequency
    """

    def __init__(
        self,
        total: int,
        desc: str = "Running",
        ncols: int | None = None,
    ) -> None:
        BasePbar.__init__(self, total, desc, ncols)
        Thread.__init__(self)

        self.stop = Event()
        self.lk = Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop.set()
        self.join()

    def run(self) -> None:
        raise NotImplementedError("Run method of threadded pbar should be overridden")

    def set_description(self, desc: str):
        with self.lk:
            return super().set_description(desc)


class LivePbar(ThreaddedPbar):
    """Creates smooth animation in the terminal, overwriting the last line"""

    pbar_style = [["\\", "|", "/", "-"], ["▖", "▘", "▝", "▗"]]

    def __init__(
        self,
        total: int,
        desc: str = "Running",
        ncols: int | None = None,
        frequency: float = 10,
        progress_style: List[str] | int | None = None,
    ) -> None:
        super().__init__(total, desc, ncols)
        self.frequency = frequency

        if progress_style is None:
            self.style = self.pbar_style[0]
        elif isinstance(progress_style, int):
            self.style = self.pbar_style[progress_style]
        else:
            self.style = progress_style

    def run(self) -> None:
        for i in itertools.cycle(self.style):
            with self.lk:
                start_str = self.make_start()

            end_str = self.make_end()

            bar_str = self.make_pbar(i, len(start_str) + len(end_str))

            print(start_str, bar_str, end_str, end="\r")

            if self.n >= self.total or self.stop.is_set():
                print()
                break

            time.sleep(1 / self.frequency)


class TimedPbar(ThreaddedPbar):
    """Log progress bar at specified intervals"""

    pbar_style = [["\\", "|", "/", "-"], ["▖", "▘", "▝", "▗"]]

    def __init__(
        self,
        total: int,
        desc: str = "Running",
        ncols: int | None = None,
        period_sec: float = 3600,
        progress_style: List[str] | int | None = None,
    ) -> None:
        super().__init__(total, desc, ncols)
        self.period_sec = period_sec

        if progress_style is None:
            self.style = self.pbar_style[0]
        elif isinstance(progress_style, int):
            self.style = self.pbar_style[progress_style]
        else:
            self.style = progress_style

    def run(self) -> None:
        for i in itertools.cycle(self.style):
            with self.lk:
                start_str = self.make_start()

            end_str = self.make_end()

            bar_str = self.make_pbar(i, len(start_str) + len(end_str))

            print(start_str, bar_str, end_str)

            if self.n >= self.total or self.stop.is_set():
                print()
                break

            time.sleep(self.period_sec)


class IntervalPbar(BasePbar):
    """Progress bar is written at an interval or fraction of progress"""

    def __init__(
        self,
        total: int,
        interval: int = -1,
        fraction: float = -1.0,
        desc: str = "Running",
        ncols: int | None = None,
    ) -> None:
        super().__init__(total, desc, ncols)
        assert -1 in {interval, fraction}, "Either interval or fraction should be set"
        self.next = 0
        if fraction != -1.0:
            assert 0 < fraction < 1, "Fraction should be sensible (0<f<1)"
            self.step = int(total * fraction)
        elif interval != -1:
            assert 0 < interval < total, "Interval should be sensible (0<i<total)"
            self.step = interval
        else:
            raise ValueError("Neither interval or fraction is specified")

    def update(self, update):
        super().update(update)
        if self.n > self.next:
            self.write()
            self.next += self.step

    def write(self):
        start_str = self.make_start()
        end_str = self.make_end()
        bar_str = self.make_pbar("", len(start_str) + len(end_str))
        print(start_str, bar_str, end_str)


def training_function(data: Any, pbar) -> None:
    for _ in data:
        pbar.update(1)
        time.sleep(0.01)


def test_progress_bar() -> None:
    data = range(1000)
    fn = pbar_wrapper(
        training_function, pbar_type=PbarType.TQDM, total=len(data), desc="tqdm"
    )
    fn(data)
    fn = pbar_wrapper(
        training_function, pbar_type=PbarType.LIVE, total=len(data), desc="live"
    )
    fn(data)
    fn = pbar_wrapper(
        training_function,
        pbar_type=PbarType.TIMED,
        total=len(data),
        desc="timed",
        period_sec=1,
    )
    fn(data)
    fn = pbar_wrapper(
        training_function,
        pbar_type=PbarType.INTERVAL,
        total=len(data),
        desc="interval",
        interval=100,
    )
    fn(data)
    fn = pbar_wrapper(
        training_function,
        pbar_type=PbarType.INTERVAL,
        total=len(data),
        desc="fraction",
        fraction=0.1,
    )
    fn(data)


if __name__ == "__main__":
    test_progress_bar()
