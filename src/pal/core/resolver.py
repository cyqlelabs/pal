"""PAL dependency resolution and import management."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from ..exceptions.core import PALCircularDependencyError, PALResolverError
from ..models.schema import ComponentLibrary, PromptAssembly
from .loader import Loader


class ResolverCache:
    """Cache for resolved dependencies to avoid redundant loading."""

    def __init__(self) -> None:
        self._cache: dict[str, ComponentLibrary] = {}

    def get(self, path_or_url: str) -> ComponentLibrary | None:
        """Get cached library by path or URL."""
        return self._cache.get(path_or_url)

    def set(self, path_or_url: str, library: ComponentLibrary) -> None:
        """Cache a loaded library."""
        self._cache[path_or_url] = library

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class DependencyGraph:
    """Tracks dependency relationships to detect cycles."""

    def __init__(self) -> None:
        self._dependencies: dict[str, set[str]] = {}
        self._visiting: set[str] = set()
        self._visited: set[str] = set()

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Add a dependency relationship."""
        if dependent not in self._dependencies:
            self._dependencies[dependent] = set()
        self._dependencies[dependent].add(dependency)

    def check_cycles(self, start: str) -> None:
        """Check for circular dependencies using DFS."""
        self._visiting.clear()
        self._visited.clear()
        self._dfs(start, [])

    def _dfs(self, node: str, path: list[str]) -> None:
        """Depth-first search for cycle detection."""
        if node in self._visiting:
            cycle_start = path.index(node)
            cycle = " -> ".join(path[cycle_start:] + [node])
            raise PALCircularDependencyError(
                f"Circular dependency detected: {cycle}",
                context={"cycle": path[cycle_start:] + [node]},
            )

        if node in self._visited:
            return

        self._visiting.add(node)
        path.append(node)

        for dependency in self._dependencies.get(node, []):
            self._dfs(dependency, path.copy())

        self._visiting.remove(node)
        self._visited.add(node)
        path.pop()


class Resolver:
    """Resolves PAL imports and manages dependency graphs."""

    def __init__(self, loader: Loader, cache: ResolverCache | None = None) -> None:
        """Initialize resolver with a loader and optional cache."""
        self.loader = loader
        self.cache = cache or ResolverCache()
        self.dependency_graph = DependencyGraph()

    async def resolve_dependencies(
        self, prompt_assembly: PromptAssembly, base_path: str | Path | None = None
    ) -> dict[str, ComponentLibrary]:
        """Resolve all dependencies for a prompt assembly.

        The dependency graph is built locally for each resolution, so concurrent
        resolutions sharing this Resolver cannot observe each other's edges.
        """
        resolved: dict[str, ComponentLibrary] = {}
        graph = DependencyGraph()

        # Resolve each import
        for alias, path_or_url in prompt_assembly.imports.items():
            resolved[alias] = await self._resolve_single_dependency(
                path_or_url, base_path, prompt_assembly.id, graph
            )

        # Check for circular dependencies
        graph.check_cycles(prompt_assembly.id)

        # Publish the graph from this resolution for introspection
        self.dependency_graph = graph

        return resolved

    async def _resolve_single_dependency(
        self,
        path_or_url: str | Path,
        base_path: str | Path | None,
        dependent_id: str,
        graph: DependencyGraph | None = None,
    ) -> ComponentLibrary:
        """Resolve a single dependency."""
        graph = graph if graph is not None else self.dependency_graph

        # Normalize path
        resolved_path = self._resolve_path(path_or_url, base_path)
        path_str = str(resolved_path)

        # Check cache first
        cached = self.cache.get(path_str)
        if cached:
            graph.add_dependency(dependent_id, cached.library_id)
            return cached

        # Load the library
        try:
            library = await self.loader.load_component_library_async(resolved_path)
        except Exception as e:
            raise PALResolverError(
                f"Failed to load dependency {path_str}: {e}{self._import_hint(path_str)}",
                context={"path": path_str, "dependent": dependent_id, "error": str(e)},
            ) from e

        # Cache the library
        self.cache.set(path_str, library)

        # Add to dependency graph
        graph.add_dependency(dependent_id, library.library_id)

        # If this library has its own imports (for future extension)
        # we would resolve them recursively here

        return library

    @staticmethod
    def _import_hint(path_str: str) -> str:
        """Explain the most common cause of a failed import."""
        if path_str.endswith(".pal") and not path_str.endswith(".pal.lib"):
            return (
                ". Imports must point at a component library (.pal.lib); "
                "prompt assemblies (.pal) cannot be imported"
            )
        return ""

    def _resolve_path(
        self, path_or_url: str | Path, base_path: str | Path | None
    ) -> str | Path:
        """Resolve relative paths relative to base_path."""
        path_str = str(path_or_url)

        # If it's a URL, return as-is
        parsed = urlparse(path_str)
        if parsed.scheme in ("http", "https"):
            return path_str

        # Convert to Path for local files
        path = Path(path_str)

        # If absolute, return as-is
        if path.is_absolute():
            return path

        # If relative and we have a base path, resolve relative to it
        if base_path:
            return Path(base_path).parent / path

        # Otherwise return as-is (will be resolved relative to current dir)
        return path

    def validate_references(
        self,
        prompt_assembly: PromptAssembly,
        resolved_libraries: dict[str, ComponentLibrary],
        variable_names: set[str] | None = None,
    ) -> list[str]:
        """Validate that all component references in composition exist.

        Args:
            prompt_assembly: The assembly whose composition is checked
            resolved_libraries: Libraries resolved for this assembly
            variable_names: Additional variable names supplied at compile time,
                which are not declared in the assembly but are still valid
        """
        errors = []

        # Variables may legitimately be accessed with dot notation (e.g. a dict
        # variable rendered as {{ config.name }}), so they are not component
        # references.
        known_names = {var.name for var in prompt_assembly.variables}
        known_names.update(variable_names or ())

        # Extract component references from composition
        component_refs = self._extract_component_references(
            prompt_assembly.composition, known_names
        )

        # Check each reference
        for ref in component_refs:
            if "." not in ref:
                errors.append(
                    f"Invalid component reference '{ref}': must be in format 'alias.component'"
                )
                continue

            alias, component_name = ref.split(".", 1)

            # Check if alias exists in imports
            if alias not in resolved_libraries:
                errors.append(f"Unknown import alias '{alias}' in reference '{ref}'")
                continue

            # Check if component exists in library
            library = resolved_libraries[alias]
            component_names = {comp.name for comp in library.components}
            if component_name not in component_names:
                errors.append(
                    f"Component '{component_name}' not found in library '{alias}'. "
                    f"Available components: {sorted(component_names)}"
                )

        return errors

    # Matches the start of a `{{ alias.component ... }}` expression, so that
    # references carrying filters or further attribute access are still seen
    # (e.g. `{{ personas.expert | trim }}`).
    _REFERENCE_PATTERN = re.compile(
        r"\{\{-?\s*([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)"
    )

    # Names bound by the template itself; `{% for a, b in ... %}` and
    # `{% set a, b = ... %}` both bind every name in the target list.
    _TARGET_LIST = r"([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)"
    _FOR_PATTERN = re.compile(r"\{%-?\s*for\s+" + _TARGET_LIST + r"\s+in\s")
    _SET_PATTERN = re.compile(r"\{%-?\s*set\s+" + _TARGET_LIST + r"\s*=")

    # Jinja2 built-ins that are never component imports
    _JINJA_BUILTINS = frozenset({"loop", "super", "self", "context", "namespace"})

    def _extract_component_references(
        self, composition: list[str], known_names: set[str] | None = None
    ) -> set[str]:
        """Extract component references from composition strings.

        Args:
            composition: The composition lines to scan
            known_names: Names that are known not to be import aliases, such as
                declared variables

        Returns:
            Set of `alias.component` references
        """
        references: set[str] = set()

        # Join all composition items to analyze loop contexts
        full_composition = "\n".join(composition)

        # Names bound by the template rather than by an import
        local_names = set(known_names or ())
        local_names.update(self._JINJA_BUILTINS)
        for match in self._FOR_PATTERN.finditer(full_composition):
            local_names.update(name.strip() for name in match.group(1).split(","))
        for match in self._SET_PATTERN.finditer(full_composition):
            local_names.update(name.strip() for name in match.group(1).split(","))

        for item in composition:
            for alias, component_name in self._REFERENCE_PATTERN.findall(item):
                if alias in local_names:
                    continue
                references.add(f"{alias}.{component_name}")

        return references

    def clear_cache(self) -> None:
        """Clear the resolver cache."""
        self.cache.clear()
