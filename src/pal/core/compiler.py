"""PAL prompt compilation with Jinja2 templating."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    StrictUndefined,
    TemplateError,
    meta,
)
from jinja2.sandbox import SandboxedEnvironment

from ..exceptions.core import (
    PALCompilerError,
    PALMissingComponentError,
    PALMissingVariableError,
)
from ..models.schema import ComponentLibrary, PALVariable, PromptAssembly, VariableType
from .loader import Loader
from .resolver import Resolver, ResolverCache


class ComponentTemplateLoader(BaseLoader):
    """Custom Jinja2 loader for PAL components.

    This loader backs the `{% include "alias.component" %}` form, which compiles
    the component as a template, so any Jinja in its content is rendered against
    the surrounding context. The `{{ alias.component }}` form is different: it
    injects the component's content as literal text and never renders it. Use
    include when a component is meant to interpolate variables.
    """

    def __init__(self, resolved_libraries: dict[str, ComponentLibrary]) -> None:
        self.resolved_libraries = resolved_libraries

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool] | None]:
        """Get template source for component references."""
        if "." not in template:
            raise TemplateError(
                f"Component reference must be in format 'alias.component', got: {template}"
            )

        alias, component_name = template.split(".", 1)

        if alias not in self.resolved_libraries:
            raise TemplateError(f"Unknown import alias: {alias}")

        library = self.resolved_libraries[alias]

        for component in library.components:
            if component.name == component_name:
                return component.content, None, lambda: True

        available = [comp.name for comp in library.components]
        raise TemplateError(
            f"Component '{component_name}' not found in library '{alias}'. Available: {available}"
        )


class PromptCompiler:
    """Compiles PAL prompt assemblies into executable prompt strings.

    The PromptCompiler is responsible for transforming PAL prompt assemblies
    into fully rendered prompt strings ready for LLM execution. It handles:

    - Template variable resolution and type checking
    - Component library imports and dependencies
    - Jinja2 template compilation with custom loaders
    - Variable validation and default value assignment

    Attributes:
        loader: The Loader instance for loading PAL files
        resolver: The Resolver instance for dependency resolution

    Example:
        >>> import asyncio
        >>> async def example():
        ...     compiler = PromptCompiler()
        ...     prompt = await compiler.compile_from_file(
        ...         Path("prompts/api_design.pal"),
        ...         variables={"api_name": "UserService", "requirements": ["REST", "JSON"]}
        ...     )
        ...     return prompt
        >>> # asyncio.run(example())  # doctest: +SKIP
    """

    def __init__(self, loader: Loader | None = None) -> None:
        """Initialize the compiler.

        Args:
            loader: Optional Loader instance. If not provided, a default Loader is created.
        """
        self.loader = loader or Loader()
        self.resolver = Resolver(self.loader, ResolverCache())

    async def compile_from_file(
        self, pal_file: str | Path, variables: dict[str, Any] | None = None
    ) -> str:
        """Compile a PAL file into a prompt string.

        Args:
            pal_file: Path to the .pal file to compile
            variables: Optional dictionary of variables to use in template rendering

        Returns:
            The compiled prompt string ready for LLM execution

        Raises:
            PALLoadError: If the file cannot be loaded
            PALMissingVariableError: If required variables are missing
            PALCompilerError: If compilation fails

        Example:
            >>> import asyncio
            >>> async def example():
            ...     compiler = PromptCompiler()
            ...     prompt = await compiler.compile_from_file(
            ...         Path("code_review.pal"),
            ...         {"language": "python", "code": "def add(a, b): return a + b"}
            ...     )
            ...     return prompt
            >>> # asyncio.run(example())  # doctest: +SKIP
        """
        prompt_assembly = await self.loader.load_prompt_assembly_async(pal_file)
        return await self.compile(prompt_assembly, variables, pal_file)

    def compile_from_file_sync(
        self, pal_file: str | Path, variables: dict[str, Any] | None = None
    ) -> str:
        """Synchronous version of compile_from_file.

        Convenience method for using the compiler in synchronous contexts.

        Args:
            pal_file: Path to the .pal file to compile
            variables: Optional dictionary of variables to use in template rendering

        Returns:
            The compiled prompt string ready for LLM execution

        Raises:
            PALLoadError: If the file cannot be loaded
            PALMissingVariableError: If required variables are missing
            PALCompilerError: If compilation fails
        """
        return asyncio.run(self.compile_from_file(pal_file, variables))

    async def compile(
        self,
        prompt_assembly: PromptAssembly,
        variables: dict[str, Any] | None = None,
        base_path: str | Path | None = None,
    ) -> str:
        """Compile a prompt assembly into a final prompt string.

        This is the core compilation method that processes a PromptAssembly object,
        resolves all dependencies, validates variables, and renders the final prompt.

        Args:
            prompt_assembly: The PromptAssembly object to compile
            variables: Dictionary of variables for template rendering
            base_path: Base path for resolving relative imports

        Returns:
            The fully compiled and rendered prompt string

        Raises:
            PALMissingComponentError: If referenced components are not found
            PALMissingVariableError: If required variables are missing
            PALCompilerError: If template compilation fails
        """
        variables = variables or {}

        # Resolve dependencies
        resolved_libraries = await self.resolver.resolve_dependencies(
            prompt_assembly, base_path
        )

        # Validate all component references exist
        validation_errors = self.resolver.validate_references(
            prompt_assembly, resolved_libraries, set(variables)
        )
        if validation_errors:
            raise PALMissingComponentError(
                f"Missing component references in {prompt_assembly.id}",
                context={"errors": validation_errors},
            )

        # Validate required variables are provided
        missing_vars = self._check_missing_variables(prompt_assembly, variables)
        if missing_vars:
            raise PALMissingVariableError(
                f"Missing required variables for {prompt_assembly.id}: {missing_vars}",
                context={"missing_variables": missing_vars},
            )

        # Type check and convert variables
        typed_variables = self._type_check_variables(prompt_assembly, variables)

        # Create Jinja2 environment
        env = self._create_jinja_environment(resolved_libraries)

        # Build context for templating
        context = self._build_template_context(resolved_libraries, typed_variables)

        # Join composition items and compile as a single template
        # This allows multi-line Jinja constructs to work properly
        full_composition = "\n".join(prompt_assembly.composition)

        try:
            template = env.from_string(full_composition)
            compiled_prompt = template.render(**context)
        except Exception as e:
            # Rendering can fail with more than TemplateError: filters and
            # expressions raise ordinary Python exceptions too. All of them are
            # reported as PALCompilerError so callers only handle PALError.
            raise PALCompilerError(
                f"Template error in composition: {e}",
                context={
                    "composition": full_composition[:500] + "..."
                    if len(full_composition) > 500
                    else full_composition,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "prompt_id": prompt_assembly.id,
                },
            ) from e
        return self._clean_compiled_prompt(compiled_prompt)

    def _check_missing_variables(
        self, prompt_assembly: PromptAssembly, provided_vars: dict[str, Any]
    ) -> list[str]:
        """Check for missing required variables."""
        missing = []
        for var_def in prompt_assembly.variables:
            if (
                var_def.required
                and var_def.name not in provided_vars
                and var_def.default is None
            ):
                missing.append(var_def.name)
        return missing

    def _type_check_variables(
        self, prompt_assembly: PromptAssembly, variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Type check and convert variables according to their definitions."""
        typed_vars = {}
        var_defs = {var.name: var for var in prompt_assembly.variables}

        # Process provided variables
        typed_vars.update(self._process_provided_variables(variables, var_defs))

        # Add defaults for missing variables
        self._add_default_variables(prompt_assembly.variables, typed_vars)

        return typed_vars

    def _process_provided_variables(
        self, variables: dict[str, Any], var_defs: dict[str, Any]
    ) -> dict[str, Any]:
        """Process and type-check provided variables."""
        typed_vars = {}

        for name, value in variables.items():
            if name in var_defs:
                var_def = var_defs[name]
                try:
                    typed_vars[name] = self._convert_variable(value, var_def.type)
                except (ValueError, TypeError) as e:
                    raise PALCompilerError(
                        f"Type error for variable '{name}': expected {var_def.type}, got {type(value).__name__}",
                        context={
                            "variable": name,
                            "expected_type": var_def.type,
                            "actual_type": type(value).__name__,
                            "value": str(value),
                        },
                    ) from e
            else:
                # Variable not defined in schema, pass through as-is
                typed_vars[name] = value

        return typed_vars

    def _add_default_variables(
        self, var_definitions: list[PALVariable], typed_vars: dict[str, Any]
    ) -> None:
        """Add default values for missing variables."""
        default_values: dict[VariableType, Any] = {
            VariableType.STRING: "",
            VariableType.LIST: [],
            VariableType.DICT: {},
            VariableType.BOOLEAN: False,
            VariableType.INTEGER: 0,
            VariableType.FLOAT: 0.0,
            # Without an entry, an unset optional would render as "None"
            VariableType.ANY: "",
        }

        for var_def in var_definitions:
            if var_def.name not in typed_vars:
                if var_def.default is not None:
                    # Declared defaults go through the same conversion as
                    # caller-supplied values so both reach the template typed.
                    try:
                        typed_vars[var_def.name] = self._convert_variable(
                            var_def.default, var_def.type
                        )
                    except (ValueError, TypeError) as e:
                        raise PALCompilerError(
                            f"Type error for default of variable '{var_def.name}': "
                            f"expected {var_def.type}, got {type(var_def.default).__name__}",
                            context={
                                "variable": var_def.name,
                                "expected_type": var_def.type,
                                "actual_type": type(var_def.default).__name__,
                                "value": str(var_def.default),
                            },
                        ) from e
                elif not var_def.required:
                    typed_vars[var_def.name] = default_values.get(var_def.type)

    def _convert_variable(self, value: Any, var_type: VariableType) -> Any:
        """Convert a variable to the specified type."""
        converters: dict[VariableType, Callable[[Any], Any]] = {
            VariableType.ANY: lambda v: v,
            VariableType.STRING: str,
            VariableType.INTEGER: self._convert_to_int,
            VariableType.FLOAT: self._convert_to_float,
            VariableType.BOOLEAN: self._convert_to_bool,
            VariableType.LIST: self._convert_to_list,
            VariableType.DICT: self._convert_to_dict,
        }

        if var_type not in converters:
            raise ValueError(f"Unknown variable type: {var_type}")

        return converters[var_type](value)

    def _convert_to_int(self, value: Any) -> int:
        """Convert value to integer."""
        if isinstance(value, bool):
            raise TypeError("Boolean cannot be converted to integer")
        return int(value)

    def _convert_to_float(self, value: Any) -> float:
        """Convert value to float."""
        if isinstance(value, bool):
            raise TypeError("Boolean cannot be converted to float")
        return float(value)

    def _convert_to_bool(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, str):
            lower_val = value.lower()
            if lower_val in ("true", "1", "yes", "on"):
                return True
            if lower_val in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"Cannot convert string '{value}' to boolean")
        return bool(value)

    def _convert_to_list(self, value: Any) -> list[Any]:
        """Convert value to list."""
        if not isinstance(value, list | tuple):
            raise TypeError(f"Expected list or tuple, got {type(value).__name__}")
        return list(value)

    def _convert_to_dict(self, value: Any) -> dict[str, Any]:
        """Convert value to dict."""
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")
        return value

    def _create_jinja_environment(
        self, resolved_libraries: dict[str, ComponentLibrary]
    ) -> Environment:
        """Create a configured Jinja2 environment.

        A sandboxed environment is used because compositions and components are
        data: they are routinely shared and may be imported over HTTP, so
        rendering them must not grant access to the Python runtime.
        """
        loader = ComponentTemplateLoader(resolved_libraries)

        return SandboxedEnvironment(
            loader=loader,
            undefined=StrictUndefined,  # Fail on undefined variables
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def _build_template_context(
        self, resolved_libraries: dict[str, ComponentLibrary], variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the context for Jinja2 templating."""
        context = variables.copy()

        # Add component access via aliases
        for alias, library in resolved_libraries.items():
            component_dict = {comp.name: comp.content for comp in library.components}
            context[alias] = component_dict

        return context

    # Fenced code blocks are captured so their contents survive cleanup verbatim
    _FENCED_BLOCK = re.compile(r"(```.*?(?:\n```|\Z))", re.DOTALL)

    def _clean_compiled_prompt(self, prompt: str) -> str:
        """Clean up the compiled prompt string.

        Excessive blank lines are collapsed everywhere except inside fenced code
        blocks, where blank lines are part of the content being shown.
        """
        segments = self._FENCED_BLOCK.split(prompt)

        # split() with one capture group alternates: text, fence, text, fence...
        cleaned = [
            segment if index % 2 else re.sub(r"\n\s*\n\s*\n+", "\n\n", segment)
            for index, segment in enumerate(segments)
        ]

        # Strip leading and trailing whitespace
        return "".join(cleaned).strip()

    def analyze_template_variables(self, prompt_assembly: PromptAssembly) -> set[str]:
        """Analyze and extract undeclared template variables from the composition.

        This method helps identify which variables are referenced in the template
        but not explicitly declared in the variables section. Useful for debugging
        and validation.

        Args:
            prompt_assembly: The PromptAssembly to analyze

        Returns:
            Set of undeclared variable names found in the composition

        Example:
            >>> import asyncio
            >>> from pal import PromptCompiler, Loader
            >>> async def example():
            ...     compiler = PromptCompiler()
            ...     loader = Loader()
            ...     assembly = await loader.load_prompt_assembly_async(Path("prompt.pal"))
            ...     undeclared = compiler.analyze_template_variables(assembly)
            ...     print(f"Undeclared variables: {undeclared}")
            >>> # asyncio.run(example())  # doctest: +SKIP
        """
        env = Environment()
        variables = set()

        # Join all composition items for proper analysis
        full_composition = "\n".join(prompt_assembly.composition)

        try:
            parsed = env.parse(full_composition)
            all_vars = meta.find_undeclared_variables(parsed)

            # Filter out known component imports and defined variables
            import_aliases = set(prompt_assembly.imports.keys())
            defined_vars = {var.name for var in prompt_assembly.variables}

            # Only return variables that aren't component imports or defined
            # variables. find_undeclared_variables reports root names only, so
            # `{{ alias.component }}` arrives here as `alias`.
            variables.update(all_vars - import_aliases - defined_vars)

        except TemplateError:
            # Fallback to item-by-item analysis if full parsing fails
            for item in prompt_assembly.composition:
                try:
                    parsed = env.parse(item)
                    variables.update(meta.find_undeclared_variables(parsed))
                except TemplateError:
                    continue

        return variables
