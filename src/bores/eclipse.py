import abc
import re
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bores.typing import T


class KeyWord(typing.Generic[T], abc.ABC):
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def parse(self, text: str, /) -> T: ...

    def extract_block(self, text: str, /) -> typing.Optional[str]:
        """
        Extract the data block following a GRDECL keyword up to its `/`
        terminator.

        Matches the keyword as a whole word (`\\b` boundaries), then captures
        everything up to (but not including) the next `/` that is either
        preceded by optional whitespace on a line of its own, or is
        whitespace-surrounded inline.

        :param text: Full comment-stripped GRDECL text.
        :param keyword: Keyword name (e.g. `"COORD"`).
        :returns: The raw data string between the keyword and its `/`
            terminator, or `None` if the keyword is absent.
        """
        pattern = re.compile(
            r"\b" + re.escape(self.name) + r"\b\s*(.*?)(?:\n\s*/\s*|\s*/\s*)",
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(text)
        return m.group(1).strip() if m else None

    def extract_blocks(self, text: str, /) -> typing.List[str]:
        """
        Extract **all** data blocks for a keyword that may appear multiple times
        (e.g. `FAULTS`, `MULTFLT`).

        :param text: Full comment-stripped GRDECL text.
        :param keyword: Keyword name.
        :returns: List of block strings, one per occurrence (may be empty).
        """
        pattern = re.compile(
            r"\b" + re.escape(self.name) + r"\b\s*(.*?)(?:\n\s*/\s*|\s*/\s*)",
            re.DOTALL | re.IGNORECASE,
        )
        return [m.group(1).strip() for m in pattern.finditer(text)]

    def tokenise(self, text: str, /) -> typing.List[str]:
        """
        Split comment-stripped GRDECL text into whitespace-separated tokens,
        expanding `N*value` repeat syntax in-place.

        Examples:

            "100*0"  -> ["0", "0", ..., "0"]  (100 times)
            "3*1.5"  -> ["1.5", "1.5", "1.5"]

        :param text: Comment-stripped GRDECL text.
        :returns: Flat list of expanded string tokens.
        """
        raw_tokens = text.split()
        expanded: typing.List[str] = []
        repeat_re = re.compile(r"^(\d+)\*(.+)$")
        for tok in raw_tokens:
            m = repeat_re.match(tok)
            if m:
                expanded.extend([m.group(2)] * int(m.group(1)))
            else:
                expanded.append(tok)
        return expanded

    def read_vector(
        self, text: str, expected: int
    ) -> typing.Optional[npt.NDArray[np.float64]]:
        """Read a vector  (length = expected)."""
        block = self.extract_block(text)
        if block is None:
            return None

        tokens = self.tokenise(block)
        if len(tokens) != expected:
            raise ValueError
        return np.array(tokens, dtype=np.float64)


_TextOrPath = typing.Union[str, bytes, Path]


class DataFile:
    DEFAULT_KEYWORDS: typing.ClassVar[typing.List[KeyWord]] = []

    __slots__ = ("_text", "_keywords", "_cache")

    def __init__(
        self,
        source: _TextOrPath,
        *,
        keywords: typing.Optional[typing.Sequence[KeyWord[typing.Any]]] = None,
        expand: bool = False,
    ) -> None:
        text = self._load(source, expand=expand)
        self._text = self._clean(text)
        self._keywords = {}
        self._cache = {}
        
        all_keywords = list(self.DEFAULT_KEYWORDS)
        if keywords:
            all_keywords.extend(keywords)
        self.add_keywords(*all_keywords)
        

    def _load(self, source: _TextOrPath, expand: bool = False) -> str: ...

    def _clean(self, text: str) -> str:
        """
        Remove `--` line comments from GRDECL text.

        :param text: Raw GRDECL text.
        :returns: Text with all `--` comments blanked out.
        """
        return re.sub(r"--[^\n]*", "", text)

    def get(
        self, name: str, /, *, use_cache: bool = True
    ) -> typing.Optional[typing.Any]:
        key = name.upper()
        if key not in self._keywords:
            return None

        if use_cache and self._cache and key in self._cache:
            return self._cache[key]

        keyword = self._keywords[key]
        value = keyword.parse(self._text)
        if use_cache:
            self._cache[key] = value
        return value

    def add_keywords(self, *keywords: KeyWord[typing.Any]) -> None:
        if not keywords:
            return
        self._keywords.update({keyword.name.upper(): keyword for keyword in keywords})
