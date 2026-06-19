"""
Top-level entry point for parsing Eclipse / GRDECL keyword decks.

Usage:

```python
from bores.eclipse import DataFile
from bores.eclipse.keywords import Poro, PermX, Faults

df = DataFile("path/to/model.DATA")
poro = df.get("PORO")    # ndarray (n_cells,) or None
faults = df.get("FAULTS")  # List[Dict] or None

# With extra keywords not in the default set:
df2 = DataFile(text, keywords=[MyCustomKeyword()])
val = df2.get("MYCUSTOM")
```
"""

import typing

from bores.eclipse.core import (
    Deck,
    GridDimensions,
    _TextOrPath,
    resolve_source,
    strip_comments,
)
from bores.eclipse.keywords.base import Keyword
from bores.eclipse.keywords.geometry import (
    Coord,
    Dimens,
    GridUnit,
    MapAxes,
    MapUnits,
    Pinch,
    SpecGrid,
    ZCorn,
)
from bores.eclipse.keywords.petrophysics import (
    ActNum,
    Dx,
    Dy,
    Dz,
    MultX,
    MultXMinus,
    MultY,
    MultYMinus,
    MultZ,
    MultZMinus,
    PermX,
    PermY,
    PermZ,
    Poro,
    Tops,
)
from bores.eclipse.keywords.structure import NNC, Faults, MultFLT

__all__ = ["DataFile"]


_DIMENSION_KEYWORDS: typing.Tuple[str, ...] = ("SPECGRID", "DIMENS")


class DataFile:
    """
    A parsed Eclipse deck: comment-stripped, `INCLUDE`-resolved text plus a
    registry of `bores.eclipse.keywords.base.Keyword` objects, exposed through `get`.

    Grid dimensions (`SPECGRID` or `DIMENS`) are resolved once, eagerly,
    at construction time, because every
    `bores.eclipse.keywords.array.GridArrayKeyword` and the
    corner-point keywords (`COORD`, `ZCORN`) need them to know expected
    array lengths and to resolve `BOX` operator scopes.

    Parsed keyword values are cached; use `use_cache=False` to force
    re-parsing (mainly useful in tests).

    **Usage**:

    ```python
    from bores.eclipse import DataFile
    from bores.eclipse.keywords import Poro, PermX, Faults

    df = DataFile("path/to/model.DATA")
    poro = df.get("PORO")    # ndarray (n_cells,) or None
    faults = df.get("FAULTS")  # List[Dict] or None

    # With extra keywords not in the default set:
    df2 = DataFile(text, keywords=[MyCustomKeyword()])
    val = df2.get("MYCUSTOM")
    ```

    **Adding a new keyword**:

    ```python
    from bores.eclipse.keywords import GridArrayKeyword

    df.add_keywords(GridArrayKeyword("MYARRAY"))
    val = df.get("MYARRAY")
    ```

    **Or pass it at construction time**:
    ```python
    df = DataFile(source, keywords=[GridArrayKeyword("MYARRAY")])
    ```

    """

    DEFAULT_KEYWORDS: typing.ClassVar[typing.List[Keyword[typing.Any]]] = [
        # Dimension
        SpecGrid(),
        Dimens(),
        # Geometry
        MapAxes(),
        GridUnit(),
        MapUnits(),
        Pinch(),
        Coord(),
        ZCorn(),
        # Geometry arrays
        Tops(),
        Dx(),
        Dy(),
        Dz(),
        ActNum(),
        # Structure
        Faults(),
        MultFLT(),
        NNC(),
        # Transmissibility multipliers
        MultX(),
        MultXMinus(),
        MultY(),
        MultYMinus(),
        MultZ(),
        MultZMinus(),
        # Petrophysics (common subset; callers can add more)
        Poro(),
        PermX(),
        PermY(),
        PermZ(),
    ]

    __slots__ = ("_deck", "_keywords", "_cache", "dimensions")

    def __init__(
        self,
        source: _TextOrPath,
        *,
        keywords: typing.Optional[typing.Sequence[Keyword[typing.Any]]] = None,
        encoding: str = "ascii",
    ) -> None:
        """
        :param source: A file path, raw deck text `str`, or raw deck
            `bytes`.  Filesystem paths and `Path` objects are read from
            disk; `INCLUDE` directives are resolved relative to the source
            file's directory.  Raw text input drops `INCLUDE` directives
            with a warning.
        :param keywords: Additional `bores.eclipse.keywords.base.Keyword`
            instances to register *in addition to* `DEFAULT_KEYWORDS`.
            Keywords with the same name override the default.
        :param encoding: Text encoding for file/bytes input (default
            `"ascii"`).
        :raises DeckParseError: If a file cannot be read.
        """
        text = resolve_source(source, encoding=encoding)
        clean_text = strip_comments(text)
        self._deck = Deck(clean_text)

        self._keywords: typing.Dict[str, Keyword[typing.Any]] = {
            kw.name: kw for kw in self.DEFAULT_KEYWORDS
        }
        if keywords:
            self._keywords.update({kw.name: kw for kw in keywords})

        self._cache: typing.Dict[str, typing.Any] = {}
        self.dimensions: typing.Optional[GridDimensions] = self._resolve_dimensions()

    def _resolve_dimensions(self) -> typing.Optional[GridDimensions]:
        """
        Resolve grid extent from `SPECGRID` (preferred) or `DIMENS`.

        :returns: A `bores.eclipse.core.GridDimensions`, or `None`
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
        """The underlying scanned `bores.eclipse.core.Deck`."""
        return self._deck

    def add_keywords(self, *keywords: Keyword[typing.Any]) -> None:
        """
        Register additional keywords (or override existing ones by name).

        Clears any cached value for a keyword name that is being replaced.

        :param keywords: One or more `bores.eclipse.keywords.base.Keyword`
            instances.
        """
        for keyword in keywords:
            self._keywords[keyword.name] = keyword
            self._cache.pop(keyword.name, None)

    def has(self, name: str) -> bool:
        """Return whether `name` occurs anywhere in the deck."""
        return self._deck.has(name.upper())

    def get(
        self, name: str, /, *, use_cache: bool = True
    ) -> typing.Optional[typing.Any]:
        """
        Parse and return the value of keyword `name`.

        :param name: Keyword name, case-insensitive.
        :param use_cache: Whether to read/write the per-`DataFile`
            value cache. Set `False` to force re-parsing (mainly for testing).
        :returns: The keyword's parsed value, or `None` if it is absent
            from the deck or not registered.
        """
        key = name.upper()
        if key not in self._keywords:
            return None
        
        if use_cache and key in self._cache:
            return self._cache[key]
        
        value = self._keywords[key].parse(self._deck, self.dimensions)
        if use_cache:
            self._cache[key] = value
        return value

    def get_many(
        self, *names: str, use_cache: bool = True
    ) -> typing.Dict[str, typing.Optional[typing.Any]]:
        """
        Convenience batch form of `get`.

        :param names: Keyword names to retrieve.
        :param use_cache: Passed through to `get`.
        :returns: `{name: value}` dict in the order requested.
        """
        return {name: self.get(name, use_cache=use_cache) for name in names}
