import os
from collections.abc import Collection, Sequence
from typing import TextIO


def printable(value: str) -> str:
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in value)


class Terminal:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.color = stream.isatty() and "NO_COLOR" not in os.environ

    def line(self, value: str = "") -> None:
        print(value, file=self.stream)

    def _style(self, value: str, code: str) -> str:
        return f"\033[{code}m{value}\033[0m" if self.color else value

    def dim(self, value: str) -> str:
        return self._style(value, "2")

    def status(self, value: str) -> str:
        if value.startswith("PASS"):
            return self._style(value, "32")
        if value.startswith("FAIL"):
            return self._style(value, "31")
        return value

    def delta(self, value: str) -> str:
        if value.lstrip().startswith("+"):
            return self._style(value, "32")
        if value.lstrip().startswith("-"):
            return self._style(value, "31")
        return self.dim(value)

    def table(
        self,
        headings: tuple[str, ...],
        rows: Sequence[tuple[str, ...]],
        *,
        right: Collection[int] = (),
        gate: bool = False,
        delta: int | None = None,
    ) -> None:
        widths = [max(map(len, column)) for column in zip(headings, *rows, strict=True)]

        def columns(row: tuple[str, ...]) -> list[str]:
            return [
                value.rjust(width) if i in right else value.ljust(width)
                for i, (value, width) in enumerate(zip(row, widths, strict=True))
            ]

        self.line(self.dim("  ".join(columns(headings)).rstrip()))
        for row in rows:
            cells = columns(row)
            cells[-1] = cells[-1].rstrip()
            if delta is not None:
                cells[delta] = self.delta(cells[delta])
            if gate:
                cells[-1] = self.status(cells[-1])
            self.line("  ".join(cells))


