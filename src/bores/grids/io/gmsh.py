"""
Gmsh `.msh` mesh reader.

Reads Gmsh ASCII mesh files (format version 2.2) and converts 3-D volumetric
elements to a `bores.grids.base.Grid` via the
`bores.grids.factories.polyhedral.make_polyhedral_grid` factory.

**Supported element types**:

=======  ========  =========
Gmsh ID  Name       Vertices
=======  ========  =========
4        tetra      4
5        hexahedron 8
6        prism      6
7        pyramid    5
=======  ========  =========

2-D surface elements (triangles, quads, lines) are silently skipped.

**References**:

Gmsh reference manual - MSH file format, v2.
"""
import re
import typing
from pathlib import Path

import numpy as np

from bores.errors import GridImportError, UnsupportedGridFormatError
from bores.grids.base import Grid
from bores.grids.factories.polyhedral import make_polyhedral_grid

__all__ = ["load_msh"]

_TextOrPath = typing.Union[str, bytes, Path]

# Gmsh element type ID -> (n_vertices, cell_type_name)
# Only 3-D volumetric elements supported for simulation grids.
_GMSH_ELEM_TYPES: typing.Dict[int, typing.Tuple[int, str]] = {
    4: (4, "tetra"),
    5: (8, "hexahedron"),
    6: (6, "wedge"),
    7: (5, "pyramid"),
}


@typing.overload
def load_msh(source: Path, *, encoding: str = ...) -> Grid: ...


@typing.overload
def load_msh(source: str, *, encoding: str = ...) -> Grid: ...


@typing.overload
def load_msh(source: bytes, *, encoding: str = ...) -> Grid: ...


def load_msh(
    source: _TextOrPath,
    *,
    encoding: str = "utf-8",
) -> Grid:
    """
    Load a Gmsh `.msh` file (ASCII format 2.2) from a path, string, or bytes.

    :param source: One of:

        - `pathlib.Path` - path to a `.msh` file.
        - `str` - filesystem path or raw `.msh` text content.
        - `bytes` - raw `.msh` content.

    :param encoding: Text encoding for `bytes` / file input (default
        `"utf-8"`).
    :returns: A fully initialised `bores.grids.base.Grid` containing
        only the 3-D volumetric elements.
    :raises GridImportError: If the file is malformed or contains no
        supported 3-D elements.
    :raises UnsupportedGridFormatError: If the Gmsh format version is not
        2.2.
    """
    text = _resolve_text_source(source, encoding=encoding)
    return _parse_msh(text)


def _resolve_text_source(source: _TextOrPath, *, encoding: str) -> str:
    """
    Coerce `source` to a plain text string.

    :param source: Path, raw string, or bytes.
    :param encoding: Byte decoding encoding.
    :returns: Raw `.msh` text.
    :raises GridImportError: If a path cannot be read.
    """
    if isinstance(source, bytes):
        return source.decode(encoding)
    if isinstance(source, Path):
        try:
            return source.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read .msh file {source!r}: {exc}") from exc
    candidate = Path(source)
    if candidate.is_file():
        try:
            return candidate.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read .msh file {source!r}: {exc}") from exc
    return source


def _extract_section(text: str, section_name: str) -> typing.Optional[str]:
    """
    Extract the body of a `$SectionName … $EndSectionName` block.

    :param text: Full `.msh` text.
    :param section_name: Section name without the `$` prefix (e.g.
        `"Nodes"`).
    :returns: Section body text, or `None` if the section is absent.
    """
    pattern = re.compile(
        r"\$" + re.escape(section_name) + r"\s*(.*?)\s*\$End" + re.escape(section_name),
        re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _parse_msh(text: str) -> Grid:
    """
    Parse a Gmsh MSH v2.2 ASCII text blob into a `bores.grids.base.Grid`.

    :param text: Raw `.msh` text.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises UnsupportedGridFormatError: If the mesh format is not version 2.
    :raises GridImportError: If the `$Nodes` or `$Elements` sections are
        malformed or no 3-D elements are found.
    """
    # $MeshFormat
    mf_section = _extract_section(text, "MeshFormat")
    if mf_section is not None:
        version_str = mf_section.split()[0]
        major = int(version_str.split(".")[0])
        if major != 2:
            raise UnsupportedGridFormatError(
                f"Only Gmsh MSH format version 2 is supported; "
                f"got version {version_str!r}."
            )

    # $Nodes
    nodes_section = _extract_section(text, "Nodes")
    if nodes_section is None:
        raise GridImportError("Gmsh .msh file is missing the $Nodes section.")
    node_lines = nodes_section.splitlines()
    try:
        n_nodes = int(node_lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise GridImportError(
            f"Cannot read node count from $Nodes section: {exc}"
        ) from exc

    # Gmsh node IDs are 1-based; we map them to 0-based indices.
    node_id_to_index: typing.Dict[int, int] = {}
    coords: typing.List[typing.Tuple[float, float, float]] = []
    for line in node_lines[1 : n_nodes + 1]:
        parts = line.split()
        if len(parts) < 4:
            raise GridImportError(f"Malformed node line: {line!r}")
        node_id = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        node_id_to_index[node_id] = len(coords)
        coords.append((x, y, z))

    if not coords:
        raise GridImportError("No nodes found in the $Nodes section.")
    vertex_coordinates = np.array(coords, dtype=np.float64)

    # $Elements
    elem_section = _extract_section(text, "Elements")
    if elem_section is None:
        raise GridImportError("Gmsh .msh file is missing the $Elements section.")
    elem_lines = elem_section.splitlines()
    try:
        n_elements = int(elem_lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise GridImportError(
            f"Cannot read element count from $Elements section: {exc}"
        ) from exc

    # Accumulate cells grouped by type
    cell_blocks_by_type: typing.Dict[str, typing.List[typing.List[int]]] = {}
    for line in elem_lines[1 : n_elements + 1]:
        parts = line.split()
        if len(parts) < 3:
            continue
        elem_type_id = int(parts[1])
        if elem_type_id not in _GMSH_ELEM_TYPES:
            continue  # skip 2-D/1-D elements silently
        
        n_verts, type_name = _GMSH_ELEM_TYPES[elem_type_id]
        n_tags = int(parts[2])
        node_start = 3 + n_tags
        node_ids = [int(p) for p in parts[node_start : node_start + n_verts]]
        if len(node_ids) != n_verts:
            raise GridImportError(
                f"Element has {len(node_ids)} node IDs but expected {n_verts} "
                f"for type {type_name!r}."
            )
        # Convert to 0-based indices
        try:
            vert_indices = [node_id_to_index[nid] for nid in node_ids]
        except KeyError as exc:
            raise GridImportError(f"Element references unknown node ID {exc}.") from exc

        cell_blocks_by_type.setdefault(type_name, []).append(vert_indices)

    if not cell_blocks_by_type:
        raise GridImportError(
            "No supported 3-D elements (tetra, hexahedron, wedge, pyramid) "
            "found in the $Elements section."
        )

    cell_blocks = [
        {"cell_type": ctype, "connectivity": np.array(cells, dtype=np.int32)}
        for ctype, cells in cell_blocks_by_type.items()
    ]

    try:
        return make_polyhedral_grid(
            vertex_coordinates=vertex_coordinates,
            cell_blocks=cell_blocks,
            metadata={"source_format": "gmsh_msh"},
        )
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from Gmsh elements: {exc}"
        ) from exc
