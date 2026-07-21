"""BORES-specific error classes."""

__all__ = [
    "BORESError",
    "ComputationError",
    "DeserializationError",
    "PreconditionerError",
    "SerializableError",
    "SerializationError",
    "SimulationError",
    "SolverError",
    "StopSimulation",
    "StorageError",
    "StreamError",
    "TimingError",
    "ValidationError",
]


class BORESError(Exception):
    """Base class for all BORES-related errors."""

    pass


class ValidationError(BORESError, ValueError):
    """Raised when input data fails validation checks."""

    pass


# Solver Errors
class PreconditionerError(BORESError):
    """Raised when there is an error related to preconditioners."""

    pass


class SolverError(BORESError):
    """Raised when a solver fails to solve the given matrix system either due to convergence or other issues."""

    pass


class ComputationError(BORESError):
    """Raised when there is an error during numerical computations."""

    pass


# Simulation Errors
class SimulationError(BORESError):
    """Base class for simulation-related errors."""

    pass


class TimingError(SimulationError):
    """Raised when there is an error related to simulation timing."""

    pass


class StopSimulation(Exception):
    """Raised to signal that the simulation should stop gracefully."""

    pass


class StreamError(SimulationError):
    """Raised when there is an error related to streaming operations."""

    pass


# Serialization Errors
class SerializableError(BORESError):
    """Raised for errors related to the `Serializable` API."""

    pass


class SerializationError(SerializableError):
    """Raised for errors related to serialization of objects."""

    pass


class DeserializationError(SerializableError):
    """Raised for errors related to deserialization of objects."""

    pass


# Storage Errors
class StorageError(BORESError):
    """Raised when there is an error related to data storage operations."""

    pass


# Gridding Errors
class GridError(BORESError):
    """
    Base exception for all grid-related errors.
    """


class InvalidGridError(GridError, ValidationError):
    """
    Raised when a grid definition is invalid.
    """


class InvalidPointArrayError(InvalidGridError):
    """
    Raised when the point coordinate array is invalid.
    """


class InvalidCellConnectivityError(InvalidGridError):
    """
    Raised when cell connectivity is invalid.
    """


class InvalidFaceConnectivityError(InvalidGridError):
    """
    Raised when face connectivity is invalid.
    """


class InvalidGeometryError(InvalidGridError):
    """
    Raised when derived geometry is inconsistent.
    """


class InvalidVolumeError(InvalidGeometryError):
    """
    Raised when one or more cells have invalid volumes.
    """


class InvalidFaceAreaError(InvalidGeometryError):
    """
    Raised when one or more faces have invalid areas.
    """


class InvalidNormalVectorError(InvalidGeometryError):
    """
    Raised when one or more face normals are invalid.
    """


class CellNotFoundError(GridError):
    """
    Raised when a requested cell does not exist.
    """


class FaceNotFoundError(GridError):
    """
    Raised when a requested face does not exist.
    """


class PointNotFoundError(GridError):
    """
    Raised when a requested point does not exist.
    """


class GridIOError(GridError):
    """
    Base exception for grid import/export failures.
    """


class GridImportError(GridIOError):
    """
    Raised when a grid cannot be imported.
    """


class GridExportError(GridIOError):
    """
    Raised when a grid cannot be exported.
    """


class UnsupportedGridFormatError(GridIOError):
    """
    Raised when a grid format is unsupported.
    """
