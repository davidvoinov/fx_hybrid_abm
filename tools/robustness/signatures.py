"""Stable provenance digests for simulations and measurement code."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import textwrap
import types


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def calibration_semantics(data):
    """Only runtime inputs, not calibration prose, define a simulation."""
    if not isinstance(data, dict):
        return data
    return {'cli_defaults': data.get('cli_defaults', {})}


def model_signature_files(root=ROOT):
    out = []
    base = os.path.join(root, 'AgentBasedModel')
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name != '__pycache__']
        for name in filenames:
            if name.endswith('.py'):
                out.append(os.path.relpath(os.path.join(dirpath, name), root))
    out.extend(('main.py', os.path.join('calibration', 'primary_model.json')))
    return tuple(sorted(out))


_MODEL_CACHE = {}


def model_signature(root=ROOT):
    """Digest only inputs that can change a simulated trajectory."""
    root = os.path.abspath(root)
    if root in _MODEL_CACHE:
        return _MODEL_CACHE[root]
    digest = hashlib.sha256()
    for relative in model_signature_files(root):
        path = os.path.join(root, relative)
        digest.update(relative.encode('utf-8'))
        try:
            if relative == os.path.join('calibration', 'primary_model.json'):
                with open(path, encoding='utf-8') as handle:
                    payload = calibration_semantics(json.load(handle))
                digest.update(json.dumps(
                    payload, sort_keys=True, separators=(',', ':')
                ).encode('utf-8'))
            else:
                with open(path, 'rb') as handle:
                    digest.update(handle.read())
        except OSError:
            digest.update(b'<missing>')
    value = digest.hexdigest()[:16]
    _MODEL_CACHE[root] = value
    return value


_INSTALLED_DIRECTORY_NAMES = frozenset(
    {'site-packages', 'dist-packages', '.tox', 'node_modules'}
)


def _is_installed_dependency(path, root):
    """True when *path* belongs to an installed package, not to the project.

    Ownership is decided by what a file is, not by where the interpreter
    happens to live. A virtual environment created inside the working tree
    (``./.venv``) puts NumPy below ``root`` and would otherwise be hashed as
    project source, which both contradicts the documented boundary and makes
    the digest depend on the install location rather than on the code.
    """
    relative = os.path.relpath(path, root)
    parts = relative.split(os.sep)
    if _INSTALLED_DIRECTORY_NAMES.intersection(parts[:-1]):
        return True
    # An environment is recognised by its own marker file rather than by a
    # conventional directory name, so ``.venv``, ``venv`` and ``env`` are all
    # covered without enumerating them.
    directory = os.path.dirname(path)
    while os.path.commonpath((root, directory)) == root and directory != root:
        if os.path.isfile(os.path.join(directory, 'pyvenv.cfg')):
            return True
        directory = os.path.dirname(directory)
    return False


def _project_source_path(value, root=ROOT):
    """Return a canonical path for a project-owned Python callable."""
    if inspect.ismethod(value):
        value = value.__func__
    try:
        value = inspect.unwrap(value)
    except (TypeError, ValueError):
        pass
    try:
        path = inspect.getsourcefile(value) or inspect.getfile(value)
    except (OSError, TypeError):
        return None
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    if not os.path.isfile(path):
        return None
    try:
        if os.path.commonpath((root, path)) != root:
            return None
    except ValueError:
        return None
    if _is_installed_dependency(path, root):
        return None
    return path


def _code_names(code):
    """Names referenced by a function, including nested code objects."""
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            names.update(_code_names(value))
    return names


def _function_dependencies(function, root=ROOT):
    """Project callables reached through globals, modules and owner classes."""
    if inspect.ismethod(function):
        function = function.__func__
    try:
        function = inspect.unwrap(function)
    except (TypeError, ValueError):
        pass
    if not inspect.isfunction(function):
        return ()

    names = _code_names(function.__code__)
    namespace = function.__globals__
    candidates = []
    for name in names:
        if name in namespace:
            candidates.append(namespace[name])

    # ``module.helper()`` stores only the module in ``__globals__``. Resolve
    # exact attribute chains from the source rather than probing every bytecode
    # name on every module (probing NumPy aliases can itself emit warnings).
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            attributes = []
            cursor = node
            while isinstance(cursor, ast.Attribute):
                attributes.append(cursor.attr)
                cursor = cursor.value
            if not isinstance(cursor, ast.Name) or cursor.id not in namespace:
                continue
            candidate = namespace[cursor.id]
            try:
                for attribute in reversed(attributes):
                    candidate = getattr(candidate, attribute)
            except (AttributeError, RuntimeError):
                continue
            candidates.append(candidate)

    # Methods commonly call sibling methods as ``self.helper()``. The helper
    # name is in the bytecode but its class is not a function global, so recover
    # the owner from the qualified name.
    owner_name = function.__qualname__.split('.<locals>', 1)[0].split('.', 1)[0]
    owner = namespace.get(owner_name)
    if inspect.isclass(owner):
        for name in names:
            try:
                candidates.append(getattr(owner, name))
            except (AttributeError, RuntimeError):
                pass

    closure = function.__closure__ or ()
    for _name, cell in zip(function.__code__.co_freevars, closure):
        try:
            candidates.append(cell.cell_contents)
        except ValueError:
            pass

    out = []
    for candidate in candidates:
        if isinstance(candidate, (staticmethod, classmethod)):
            candidate = candidate.__func__
        if inspect.ismethod(candidate):
            candidate = candidate.__func__
        if (inspect.isfunction(candidate) or inspect.isclass(candidate)) \
                and _project_source_path(candidate, root) is not None:
            out.append(candidate)
    return tuple(out)


def transitive_source_digest(functions=(), root=ROOT, excluded_paths=()):
    """Hash the project-owned source closure of ``functions``.

    The old implementation hashed only each explicitly supplied function body.
    A change to a locally imported helper therefore left the parent signature
    untouched. This closure follows project functions through globals, module
    aliases, closures and sibling methods. External libraries remain outside
    the source digest; their effects enter through recorded inputs and the
    environment, not unstable installation paths.

    Missing source for an explicitly supplied callable is an error. Silently
    replacing it with one common marker would make unrelated unknown programs
    share a publication signature, which is not fail closed.
    """
    root = os.path.realpath(root)
    excluded = {os.path.normpath(str(path)) for path in excluded_paths}
    pending = list(functions)
    records = {}
    while pending:
        value = pending.pop()
        if inspect.ismethod(value):
            value = value.__func__
        try:
            value = inspect.unwrap(value)
        except (TypeError, ValueError):
            pass
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            raise TypeError(
                'signature roots must be Python functions, methods or classes'
            )
        path = _project_source_path(value, root)
        if path is None:
            raise ValueError(
                f'signature root {value!r} has no source below {root}'
            )
        # The path below the project root and the qualified name identify the
        # function. ``__module__`` is not part of the key, because it is not a
        # property of the code: a module run as ``python -m pkg.mod`` reports
        # ``__main__`` where the same file imported as ``pkg.mod`` reports its
        # dotted name, so including it made one unchanged source tree produce
        # two different signatures depending on how the process was started.
        # A protocol frozen from an import then refused the identical code run
        # from the command line, which is the opposite of what a code signature
        # is for.
        key = (
            os.path.relpath(path, root),
            getattr(value, '__qualname__', getattr(value, '__name__', '')),
        )
        # Simulation files are already represented by ``model_digest``. Do
        # not read and hash them a second time: apart from needless work, a
        # source-tree edit during a long run could otherwise mix the model
        # snapshot taken at launch with a later on-disk helper snapshot.
        if os.path.normpath(key[0]) in excluded:
            continue
        if key in records:
            continue
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f'cannot inspect signature dependency {key!r}'
            ) from exc
        records[key] = source

        if inspect.isclass(value):
            for member in vars(value).values():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    for accessor in (member.fget, member.fset, member.fdel):
                        if accessor is not None:
                            pending.append(accessor)
                    continue
                if ((inspect.isfunction(member) or inspect.ismethod(member))
                        and _project_source_path(member, root) is not None):
                    pending.append(member)
        else:
            pending.extend(_function_dependencies(value, root))

    digest = hashlib.sha256()
    digest.update(b'transitive-project-source-v1\0')
    for key in sorted(records):
        digest.update(json.dumps(key, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\0')
        digest.update(records[key].encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def measurement_signature(name, model_digest, functions=(), constants=()):
    """Digest model, constants and the transitive measurement source graph."""
    digest = hashlib.sha256()
    digest.update(b'measurement-signature-v2\0')
    digest.update(str(name).encode('utf-8'))
    digest.update(str(model_digest).encode('utf-8'))
    digest.update(repr(tuple(constants)).encode('utf-8'))
    digest.update(transitive_source_digest(
        functions,
        ROOT,
        excluded_paths=model_signature_files(ROOT),
    ).encode('ascii'))
    return digest.hexdigest()[:16]
