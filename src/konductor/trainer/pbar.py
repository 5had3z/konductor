import time
import os
import itertools
from threading import Thread, Event
from datetime import timedelta
from typing import Any, List
import enum

from tqdm.auto import tqdm
from colorama import Fore


class PbarType(enum.Enum):
    TQDM = enum.auto()
    INBUILT = enum.auto()


def pbar_wrapper(
    func, pbar_type: PbarType = PbarType.INBUILT, *pbar_args, **pbar_kwargs
):
    """Function must take pbar as a key-word argument"""
    pbar_t = {PbarType.INBUILT: ProgressBar, PbarType.TQDM: tqdm}[pbar_type]

    def with_pbar(*args, **kwargs):
        with pbar_t(*pbar_args, **pbar_kwargs) as pbar:
            func(*args, **kwargs, pbar=pbar)

    return with_pbar


class ProgressBar(Thread):
    """
    Threadded progress bar so you get smooth animation while iterating.
    Use context manager if possible to ensure that thread is cleaned up
    even if the total is not reached.
    """

    pbar_style = [["\\", "|", "/", "-"], ["▖", "▘", "▝", "▗"]]

    def __init__(
        self,
        total: int,
        ncols: int | None = None,
        frequency: float = 10,
        progress_style: List[str] | int | None = None,
    ) -> None:
        super().__init__()
        self.total = total
        self.ncols = ncols
        self.iter = 0
        self.frequency = frequency
        self.s_time = time.time()
        self.stop = Event()

        if progress_style is None:
            self.style = self.pbar_style[0]
        elif isinstance(progress_style, int):
            self.style = self.pbar_style[progress_style]
        else:
            self.style = progress_style

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop.set()
        self.join()

    def elapsed(self) -> timedelta:
        """Elapsed time"""
        return timedelta(seconds=time.time() - self.s_time)

    def elapsed_str(self) -> str:
        """Elapsed time string with microseconds removed"""
        return str(self.elapsed()).split(".")[0]

    def estimated(self) -> timedelta:
        """Estimated completion time"""
        if self.iter == 0:
            return timedelta(hours=999)
        time_per_iter = self.elapsed() / self.iter
        return time_per_iter * (self.total - self.iter)

    def estimated_str(self) -> str:
        """Estimated completion time string with microseconds removed"""
        return str(self.estimated()).split(".")[0]

    def run(self) -> None:
        n_digits = len(str(self.total))

        for i in itertools.cycle(self.style):
            start_str = f"\r{Fore.GREEN}{self.iter:0{n_digits}}{Fore.YELLOW}/{Fore.RED}{self.total}{Fore.RESET}"
            end_str = (
                f"Elapsed: {Fore.YELLOW}{self.elapsed_str()}{Fore.RESET} "
                f"Est: {Fore.YELLOW}{self.estimated_str()}{Fore.RESET}"
            )

            ncols = os.get_terminal_size().columns if self.ncols is None else self.ncols
            # not sure why doesn't fill full console
            ncols -= len(start_str) + len(end_str) + 1
            done_bars = ncols * self.iter / self.total

            bar_str = f"{Fore.GREEN}{'█'*int(done_bars)}{Fore.YELLOW}{i}{Fore.RED}{'-'*int(ncols - done_bars)}{Fore.RESET}"

            print(start_str, bar_str, end_str, end="")

            if self.iter >= self.total or self.stop.is_set():
                print()
                break

            time.sleep(1 / self.frequency)

    def update(self, update):
        self.iter += update


def training_function(data: Any, pbar) -> None:
    for _ in data:
        pbar.update(1)
        time.sleep(0.01)


def test_progress_bar() -> None:
    data = range(100)
    fn = pbar_wrapper(training_function, pbar_type=PbarType.TQDM, total=len(data))
    fn(data)
    fn = pbar_wrapper(training_function, pbar_type=PbarType.INBUILT, total=len(data))
    fn(data)


if __name__ == "__main__":
    test_progress_bar()
