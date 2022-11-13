import time
import itertools
from colorama import Fore
from threading import Thread, Event
from datetime import timedelta


class ProgressBar(Thread):
    def __init__(self, max_iter: int, frequency: float = 10) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.iter = 0
        self.frequency = frequency
        self.s_time = time.time()
        self.stop = Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop.set()

    def elapsed(self) -> timedelta:
        """Elapsed time"""
        return timedelta(seconds=time.time() - self.s_time)

    def elapsed_str(self) -> str:
        """Elapsed time string with microseconds removed"""
        return str(self.elapsed()).split(".")[0]

    def estimated(self) -> timedelta:
        """Estimated completion time"""
        time_per_iter = self.elapsed() / self.iter
        return time_per_iter * (self.max_iter - self.iter)

    def estimated_str(self) -> str:
        """Estimated completion time string with microseconds removed"""
        return str(self.estimated()).split(".")[0]

    def run(self) -> None:
        for i in itertools.cycle(["\\", "|", "/", "-"]):
            print(
                f"\r{Fore.GREEN}{self.iter}{Fore.YELLOW}/{Fore.RED}{self.max_iter}{Fore.RESET} {i} ",
                end="",
            )

            print(f"Elapsed: {Fore.YELLOW}{self.elapsed_str()}{Fore.RESET} ", end="")
            print(f"Est: {Fore.YELLOW}{self.estimated_str()}{Fore.RESET}", end="")

            if self.iter >= self.max_iter or self.stop.is_set():
                print()
                break

            time.sleep(1 / self.frequency)

    def update(self, update):
        self.iter += update


def test_progress_bar() -> None:
    with ProgressBar(1000) as pbar:
        for _ in range(1000):
            pbar.update(1)
            time.sleep(0.01)


if __name__ == "__main__":
    test_progress_bar()
