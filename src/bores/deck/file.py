"""Entry point for parsing Eclipse / GRDECL decks."""

import typing
from collections.abc import Collection

from typing_extensions import TypeVar

from bores.deck.core import (
    Deck,
    GridDimensions,
    _TextOrPath,
    resolve_source,
    strip_comments,
)
from bores.deck.keywords.base import Keyword
from bores.deck.keywords.grid import (
    ACTNUM,
    COORD,
    DIMENS,
    DX,
    DXV,
    DY,
    DYV,
    DZ,
    DZV,
    FAULTS,
    GRIDUNIT,
    MAPAXES,
    MAPUNIT,
    MAPUNITS,
    MULTFLT,
    MULTX,
    MULTX_MINUS,
    MULTY,
    MULTY_MINUS,
    MULTZ,
    MULTZ_MINUS,
    NNC,
    NTG,
    PERMX,
    PERMY,
    PERMZ,
    PINCH,
    PINCHOUT,
    PORO,
    PORV,
    SPECGRID,
    TOPS,
    ZCORN,
)
from bores.deck.keywords.props import (
    DENSITY,
    PVCO,
    PVDG,
    PVDO,
    PVTG,
    PVTO,
    PVTW,
    ROCK,
    ROCKTAB,
    SGFN,
    SGOF,
    SOF2,
    SOF3,
    SWFN,
    SWOF,
)
from bores.deck.keywords.regions import EQLNUM, FIPNUM, IMBNUM, PVTNUM, ROCKNUM, SATNUM
from bores.deck.keywords.runspec import (
    DISGAS,
    EQLDIMS,
    FIELD,
    GAS,
    LAB,
    METRIC,
    NOSIM,
    OIL,
    REGDIMS,
    START,
    TABDIMS,
    TITLE,
    UNIFIN,
    UNIFOUT,
    VAPOIL,
    WATER,
    WELLDIMS,
)
from bores.deck.keywords.schedule import (
    COMPDAT,
    DATES,
    GCONINJE,
    GCONPROD,
    GRUPTREE,
    TSTEP,
    WCONINJE,
    WCONPROD,
    WECON,
    WELOPEN,
    WELSPECS,
    WELTARG,
    WPIMULT,
    WTEST,
)
from bores.deck.keywords.solution import (
    EQUIL,
    PRESSURE,
    RESTART,
    RS,
    RTEMP,
    RV,
    SGAS,
    SOIL,
    SWAT,
    TEMPVD,
)
from bores.deck.keywords.summary import (
    FGPR,
    FGPT,
    FOPR,
    FOPT,
    FWPR,
    FWPT,
    RGIP,
    ROIP,
    RPTRST,
    RPTSCHED,
    RWIP,
    WBHP,
    WGPR,
    WOPR,
    WTHP,
    WWPR,
)
from bores.deck.operators import Operation, resolve_operations

__all__ = ["DeckFile"]

T = TypeVar("T")

_DIMENSION_KEYWORDS: typing.Tuple[str, ...] = ("SPECGRID", "DIMENS")

DEFAULT_KEYWORDS: typing.FrozenSet[Keyword[typing.Any]] = frozenset(
    {
        ### RUNSPEC ###
        DISGAS,
        EQLDIMS,
        FIELD,
        GAS,
        LAB,
        METRIC,
        NOSIM,
        OIL,
        REGDIMS,
        START,
        TABDIMS,
        TITLE,
        UNIFIN,
        UNIFOUT,
        VAPOIL,
        WATER,
        WELLDIMS,
        ### GRID ###
        ACTNUM,
        COORD,
        DIMENS,
        DX,
        DXV,
        DY,
        DYV,
        DZ,
        DZV,
        FAULTS,
        GRIDUNIT,
        MAPAXES,
        MAPUNIT,
        MAPUNITS,
        MULTFLT,
        MULTX,
        MULTX_MINUS,
        MULTY,
        MULTY_MINUS,
        MULTZ,
        MULTZ_MINUS,
        NNC,
        NTG,
        PERMX,
        PERMY,
        PERMZ,
        PINCH,
        PINCHOUT,
        PORO,
        PORV,
        SPECGRID,
        TOPS,
        ZCORN,
        ### REGIONS ###
        EQLNUM,
        FIPNUM,
        IMBNUM,
        PVTNUM,
        ROCKNUM,
        SATNUM,
        ### PROPS ###
        DENSITY,
        PVCO,
        PVDG,
        PVDO,
        PVTG,
        PVTO,
        PVTW,
        ROCK,
        ROCKTAB,
        SGFN,
        SGOF,
        SOF2,
        SOF3,
        SWFN,
        SWOF,
        ### SOLUTION ###
        EQUIL,
        PRESSURE,
        RESTART,
        RS,
        RV,
        SGAS,
        SOIL,
        SWAT,
        RTEMP,
        TEMPVD,
        ### SCHEDULE ###
        COMPDAT,
        DATES,
        GCONINJE,
        GCONPROD,
        GRUPTREE,
        TSTEP,
        WECON,
        WELOPEN,
        WELSPECS,
        WELTARG,
        WCONINJE,
        WCONPROD,
        WPIMULT,
        WTEST,
        ### SUMMARY ###
        FGPR,
        FGPT,
        FOPR,
        FOPT,
        FWPR,
        FWPT,
        RGIP,
        ROIP,
        RPTRST,
        RPTSCHED,
        RWIP,
        WBHP,
        WGPR,
        WOPR,
        WTHP,
        WWPR,
    }
)


class DeckFile:
    """
    A parsed Eclipse deck file: comment-stripped, `INCLUDE`-resolved text plus a
    registry of `bores.deck.keywords.base.Keyword` objects, exposed through `get`.

    **Usage**:

    ```python
    from bores.deck import DeckFile, DEFAULT_KEYWORDS

    df = DeckFile("path/to/model.DATA")
    poro = df.get("PORO")    # ndarray (n_cells,) or None
    faults = df.get("FAULTS")  # List[Dict] or None

    # With extra keywords not in the default set:
    df2 = DeckFile(text, keywords=DEFAULT_KEYWORDS | [MyCustomKeyword()])
    val = df2.get("MYCUSTOM")
    ```

    **Adding a new keyword**:

    ```python
    from bores.deck.keywords.base import GridArrayKeyword

    df.add_keywords(GridArrayKeyword("MYARRAY"))
    val = df.get("MYARRAY")
    ```

    **Or pass it at construction time**:
    ```python
    df = DeckFile(source, keywords=[GridArrayKeyword("MYARRAY")])
    ```
    """

    __slots__ = ("_deck", "_keywords", "_cache", "_operations", "dimensions")

    def __init__(
        self,
        source: _TextOrPath,
        *,
        keywords: Collection[Keyword[typing.Any]] = DEFAULT_KEYWORDS,
        encoding: str = "ascii",
    ) -> None:
        """
        :param source: A file path, raw deck text `str`, or raw deck
            `bytes`. Filesystem paths and `Path` objects are read from
            disk; `INCLUDE` directives are resolved relative to the source
            file's directory. Raw text input drops `INCLUDE` directives with a warning.
        :param keywords: Supported `bores.deck.keywords.base.Keyword`
            instances. Only supported keywords can be read from the file.
        :param encoding: Text encoding for file/bytes input (default `"ascii"`).
        :raises DeckParseError: If a file cannot be read.
        """
        text = resolve_source(source, encoding=encoding)
        clean_text = strip_comments(text)
        self._deck = Deck(clean_text)

        self._keywords: typing.Dict[str, Keyword[typing.Any]] = {
            kw.name: kw for kw in keywords
        }
        self._cache: typing.Dict[str, typing.Any] = {}
        self.dimensions: typing.Optional[GridDimensions] = self._resolve_dimensions()
        self._operations: typing.Optional[typing.List[Operation]] = (
            resolve_operations(self._deck, self.dimensions)
            if self.dimensions is not None
            else None
        )

    def _resolve_dimensions(self) -> typing.Optional[GridDimensions]:
        """
        Resolve grid extent from `SPECGRID` (preferred) or `DIMENS`.

        :returns: A `bores.deck.core.GridDimensions`, or `None`
            if neither keyword is present (decks consisting only of e.g.
            PVT / relperm tables).
        """
        for name in _DIMENSION_KEYWORDS:
            record = self._deck.first_record_for(name)
            if record is None:
                continue
            keyword = self._keywords.get(name)
            if keyword is None:
                continue
            parsed = keyword.parse(self._deck, None)
            if parsed is None:
                continue
            return GridDimensions(
                nx=int(parsed["nx"]),
                ny=int(parsed["ny"]),
                nz=int(parsed["nz"]),
            )
        return None

    @property
    def deck(self) -> Deck:
        """The underlying scanned `bores.deck.core.Deck`."""
        return self._deck

    def add_keywords(self, *keywords: Keyword[typing.Any]) -> None:
        """
        Register additional keywords (or override existing ones by name).

        Clears any cached value for a keyword name that is being replaced.

        :param keywords: One or more `bores.deck.keywords.base.Keyword`
            instances.
        """
        for keyword in keywords:
            self._keywords[keyword.name] = keyword
            self._cache.pop(keyword.name, None)

    def has(self, name: str) -> bool:
        """Return whether `name` occurs anywhere in the deck."""
        return self._deck.has(name.upper())

    @typing.overload
    def get(self, k: Keyword[T], /, *, use_cache: bool = ...) -> typing.Optional[T]: ...

    @typing.overload
    def get(
        self, k: str, /, *, use_cache: bool = ...
    ) -> typing.Optional[typing.Any]: ...

    def get(
        self, k: typing.Union[str, Keyword[typing.Any]], /, *, use_cache: bool = False
    ) -> typing.Optional[typing.Any]:
        """
        Parse and return the value of keyword `k`.

        :param k: Keyword `k`, case-insensitive or a `Keyword` object (never cached if `Keyword` object).
        :param use_cache: Whether to read/write the per-`DeckFile`
            value cache. Set `False` to force re-parsing (mainly for testing).
        :returns: The keyword's parsed value, or `None` if it is absent
            from the deck or not registered.
        """
        if isinstance(k, Keyword):
            return k.parse(
                self._deck,
                self.dimensions,
                operations=self._operations,
            )

        key = k.upper()
        if key not in self._keywords:
            return None

        if use_cache and key in self._cache:
            return self._cache[key]

        value = self._keywords[key].parse(
            self._deck,
            self.dimensions,
            operations=self._operations,
        )
        if use_cache:
            self._cache[key] = value
        return value

    def gets(
        self, *ks: typing.Union[str, Keyword[typing.Any]], use_cache: bool = False
    ) -> typing.Dict[str, typing.Optional[typing.Any]]:
        """
        Convenience batch form of `get`.

        :param ks: Keyword ks to retrieve.
        :param use_cache: Passed through to `get`.
        :returns: `{name: value}` dict in the order requested.
        """
        return {
            k.name if isinstance(k, Keyword) else k: self.get(k, use_cache=use_cache)
            for k in ks
        }
