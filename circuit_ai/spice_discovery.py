"""Locate an external SPICE executable without requiring it on ``PATH``.

The project reported ``ngspice`` as unavailable for a long time because it looked
only at ``PATH``. It was installed all along, in a location that is normal for
bundled copies: as a component of another application (Autodesk Eagle ships one,
KiCad ships ``ngspice.dll``, EasyEDA ships a simulator that links it). Those
copies work for batch invocation, so refusing to look for them produced a
conclusion -- "no external verification is possible" -- that was simply wrong.

Discovery is deliberately explicit about its order, because a silently chosen
binary changes what "verified" means:

1. an explicit path the caller gave;
2. ``LATTICE_NGSPICE`` (also ``CIRCUIT_AI_NGSPICE``) in the environment;
3. ``PATH``;
4. a short list of known install and bundle locations.

Nothing is executed to decide: a candidate counts only if the file exists and is
a regular file. A version probe is a separate, optional call, because running a
GUI-capable binary with an argument it does not understand opens a window instead
of printing anything.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = [
    "ENVIRONMENT_VARIABLES",
    "MISSING_NAME_PREFIXES",
    "candidate_paths",
    "describe_search",
    "find_executable",
    "is_deliberately_missing",
    "resolve",
]


#: Environment variables checked, in order, before ``PATH``.
ENVIRONMENT_VARIABLES: tuple[str, ...] = ("LATTICE_NGSPICE", "CIRCUIT_AI_NGSPICE")


#: A caller that names an executable like this is asking for "no external
#: backend", not for a file. Several tests and specs use such a name to force the
#: unavailable path, and discovery must never override that intent -- otherwise a
#: machine that happens to have ngspice installed would turn a deliberate
#: availability test into a false pass.
MISSING_NAME_PREFIXES: tuple[str, ...] = ("definitely_missing",)


def is_deliberately_missing(name: str | Path) -> bool:
    """Whether *name* is a request for absence rather than a path."""

    text = str(name).strip().casefold()
    return any(text.startswith(prefix) for prefix in MISSING_NAME_PREFIXES)


def _names_for(name: str) -> tuple[str, ...]:
    """File names to look for, given a bare program name.

    ``shutil.which`` appends the platform executable extensions itself, but a
    direct path lookup does not, so a search written as ``Path(dir) / "ngspice"``
    silently misses ``ngspice.exe`` on Windows.
    """

    if os.name != "nt" or Path(name).suffix:
        return (name,)
    return tuple(f"{name}{suffix}" for suffix in (".exe", ".bat", ".cmd", ".com"))


#: Locations a bundled or installed copy is commonly found at. Kept short and
#: readable on purpose: a deep recursive scan of whole drives is slow and would
#: make availability depend on machine layout in ways hard to explain.
def _known_locations(name: str) -> tuple[Path, ...]:
    if os.name == "nt":
        roots = [
            Path(r"C:\Program Files"),
            Path(r"C:\Program Files (x86)"),
            Path.home() / "AppData" / "Local",
        ]
        relatives = (
            Path("Spice64") / "bin",
            Path("ngspice") / "bin",
            Path("ngspice"),
            Path("KiCad") / "bin",
            Path("lceda-pro-sim"),
        )
    else:
        roots = [Path("/usr"), Path("/usr/local"), Path("/opt")]
        relatives = (Path("bin"), Path("local") / "bin")

    names = _names_for(name)
    found: list[Path] = []
    eagle = Path("Autodesk") / "webdeploy" / "production"
    for root in roots:
        candidate = root / eagle
        if candidate.is_dir():
            # Eagle keeps the binary under a version-hash directory, and the
            # hash differs per install, so this one path has to be globbed.
            for file_name in names:
                found.extend(
                    sorted(
                        candidate.glob(
                            f"*/Applications/Electron/LibEagle/ngspice/bin/{file_name}"
                        )
                    )
                )
        for relative in relatives:
            for file_name in names:
                found.append(root / relative / file_name)
    return tuple(found)


def candidate_paths(name: str = "ngspice") -> tuple[Path, ...]:
    """Every path ``find_executable`` would consider, in order."""

    candidates: list[Path] = []
    for variable in ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    which = shutil.which(name)
    if which:
        candidates.append(Path(which))
    candidates.extend(_known_locations(name))
    # De-duplicate while preserving order, so the report reads like the search.
    seen: set[str] = set()
    ordered: list[Path] = []
    for item in candidates:
        key = str(item).casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return tuple(ordered)


def resolve(
    name: str | Path = "ngspice",
    *,
    explicit: str | Path | None = None,
) -> tuple[str, str]:
    """A usable executable together with the reason, for an availability report.

    Returns the name to invoke -- which may be the value that was passed in, when
    it is already usable -- and a sentence naming what was searched.  The reason
    is returned even on success, because "which binary answered" is part of what
    makes a verification claim checkable.
    """

    requested = str(explicit if explicit is not None else name)

    if is_deliberately_missing(requested):
        return requested, (
            f"{requested!r} requests an absent backend by convention; "
            "discovery was not attempted"
        )

    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return str(path), f"using the explicit path {path}"
        return requested, f"explicit path {path} does not exist"

    found = find_executable(requested)
    if found is not None:
        if found.name == requested and not found.is_absolute():
            return str(found), f"using {found} from PATH"
        return str(found), f"discovered at {found}"
    return requested, f"not found; {describe_search(requested)}"


def find_executable(
    name: str = "ngspice",
    *,
    explicit: str | Path | None = None,
) -> Path | None:
    """First usable executable, or ``None``.

    A candidate must be an existing regular file. It is not executed here: a
    GUI-capable binary invoked with an argument it does not understand opens a
    window rather than reporting anything, which is a bad way to answer a
    question this cheap.
    """

    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in candidate_paths(name):
        if candidate.is_file():
            return candidate
    return None


def describe_search(name: str = "ngspice", *, explicit: str | Path | None = None) -> str:
    """A human-readable account of what was tried, for an availability message.

    Availability messages are read by people deciding whether a result is
    trustworthy, so the message names the places that were searched rather than
    only reporting failure.
    """

    if explicit is not None:
        path = Path(explicit)
        return f"explicit path {path} ({'exists' if path.is_file() else 'missing'})"
    tried = candidate_paths(name)
    listed = ", ".join(str(item) for item in tried) if tried else "(nothing to try)"
    return f"searched {len(tried)} locations: {listed}"
