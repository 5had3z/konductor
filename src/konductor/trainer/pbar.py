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

    # The amount of the terminal string that's just formatting
    # This increases the "string length" but not its display length
    # So it has to be compensated for
    _fmt_len = len(
        f"{Fore.GREEN}{Fore.YELLOW}{Fore.RED}{Fore.RESET}"
        f"{Fore.YELLOW}{Fore.RESET}{Fore.YELLOW}{Fore.RESET}"
        f"{Fore.GREEN}{Fore.YELLOW}{Fore.RED}{Fore.RESET}"
    )

    def __init__(
        self,
        total: int,
        desc: str = "",
        ncols: int | None = None,
        frequency: float = 10,
        progress_style: List[str] | int | None = None,
    ) -> None:
        super().__init__()
        self.total = total
        self.ncols = ncols
        self._desc = desc
        self.n = 0
        self.frequency = frequency
        self.s_time = time.time()
        self.stop = Event()
        self.lk = Lock()

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
        if self.n == 0:
            return timedelta(hours=999)
        time_per_iter = self.elapsed() / self.n
        return time_per_iter * (self.total - self.n)

    def estimated_str(self) -> str:
        """Estimated completion time string with microseconds removed"""
        return str(self.estimated()).split(".")[0]

    def run(self) -> None:
        n_digits = len(str(self.total))

        for i in itertools.cycle(self.style):
            with self.lk:
                start_str = f"{Fore.BLUE}{self._desc}: {Fore.GREEN}{self.n:0{n_digits}}{Fore.YELLOW}/{Fore.RED}{self.total}{Fore.RESET}"
            end_str = (
                f"Elapsed: {Fore.YELLOW}{self.elapsed_str()}{Fore.RESET} "
                f"Est: {Fore.YELLOW}{self.estimated_str()}{Fore.RESET}"
            )

            ncols = os.get_terminal_size().columns if self.ncols is None else self.ncols
            print(" " * (ncols - 2), end="\r")  # Clear the terminal
            ncols -= len(start_str) + len(end_str) - self._fmt_len // 2
            done_bars = ncols * self.n // self.total

            bar_str = f"{Fore.GREEN}{'█'*done_bars}{Fore.YELLOW}{i}{Fore.RED}{'-'*(ncols - done_bars)}{Fore.RESET}"

            print(start_str, bar_str, end_str, end="\r")

            if self.n >= self.total or self.stop.is_set():
                print()
                break

            time.sleep(1 / self.frequency)

    def set_description(self, desc: str):
        with self.lk:
            self._desc = desc

    def update(self, update):
        self.n += update


def training_function(data: Any, pbar) -> None:
    for _ in data:
        pbar.update(1)
        time.sleep(0.01)


def test_progress_bar() -> None:
    data = range(100)
    fn = pbar_wrapper(
        training_function, pbar_type=PbarType.TQDM, total=len(data), desc="test"
    )
    fn(data)
    fn = pbar_wrapper(
        training_function, pbar_type=PbarType.INBUILT, total=len(data), desc="test"
    )
    fn(data)


if __name__ == "__main__":
    test_progress_bar()
