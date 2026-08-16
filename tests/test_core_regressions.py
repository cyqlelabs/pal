"""Regression tests for defects fixed in the PAL core engine.

Each test here pins the behaviour of a specific defect so it cannot come back.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from pal.core.compiler import PromptCompiler
from pal.core.evaluation import (
    ContainsAssertion,
    EvaluationRunner,
    LengthAssertion,
    RegexMatchAssertion,
)
from pal.core.executor import MockLLMClient, PromptExecutor
from pal.core.loader import Loader
from pal.exceptions.core import (
    PALCompilerError,
    PALLoadError,
    PALMissingComponentError,
    PALResolverError,
)
from pal.models.schema import PromptAssembly


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def make_assembly(composition, variables=None, imports=None, prompt_id="regression"):
    """Build a PromptAssembly from a composition and optional declarations."""
    return PromptAssembly.model_validate(
        {
            "pal_version": "1.0",
            "id": prompt_id,
            "version": "1.0.0",
            "description": "Regression fixture",
            "variables": variables or [],
            "imports": imports or {},
            "composition": composition,
        }
    )


def write_library(path, library_id, components):
    """Write a component library to disk."""
    path.write_text(
        yaml.dump(
            {
                "pal_version": "1.0",
                "library_id": library_id,
                "version": "1.0.0",
                "description": "Regression library",
                "type": "persona",
                "components": [
                    {"name": name, "description": "d", "content": content}
                    for name, content in components.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestTemplateSandbox:
    """Compositions and components must not reach the Python runtime."""

    @pytest.mark.parametrize(
        "payload",
        [
            "{{ ''.__class__.__mro__[1].__subclasses__() }}",
            "{{ ''|attr('__class__') }}",
            "{{ (1).__class__.__base__.__subclasses__() }}",
            "{{ self.__init__.__globals__ }}",
        ],
    )
    @pytest.mark.asyncio
    async def test_composition_cannot_escape_the_sandbox(self, payload):
        """Attribute traversal into Python internals is refused."""
        compiler = PromptCompiler()

        with pytest.raises(PALCompilerError) as exc_info:
            await compiler.compile(make_assembly([payload]))

        assert "unsafe" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_included_component_cannot_escape_the_sandbox(self, temp_dir):
        """A shared library component is sandboxed when included as a template."""
        lib = write_library(
            temp_dir / "evil.pal.lib",
            "evil",
            {"payload": "{{ cycler.__init__.__globals__.os.popen('id').read() }}"},
        )
        assembly = make_assembly(
            ['{% include "lib.payload" %}'], imports={"lib": str(lib)}
        )

        with pytest.raises(PALCompilerError) as exc_info:
            await PromptCompiler().compile(assembly)

        assert "unsafe" in str(exc_info.value)


class TestConcurrentResolution:
    """A shared compiler must not leak dependency state between compilations."""

    @pytest.mark.asyncio
    async def test_concurrent_compiles_do_not_report_a_false_cycle(self, temp_dir):
        """Independent compilations sharing a compiler must both succeed."""
        write_library(temp_dir / "alpha.pal.lib", "alpha", {"c": "A"})
        write_library(temp_dir / "beta.pal.lib", "beta", {"c": "B"})

        # Prompt "alpha" imports library "beta" and vice versa. Neither depends
        # on the other, but a shared graph used to merge them into a cycle.
        compiler = PromptCompiler()
        assemblies = [
            make_assembly(
                ["{{ lib.c }}"],
                imports={"lib": str(temp_dir / f"{lib}.pal.lib")},
                prompt_id=prompt_id,
            )
            for prompt_id, lib in (("alpha", "beta"), ("beta", "alpha"))
        ]

        results = await asyncio.gather(
            *(compiler.compile(assembly) for assembly in assemblies)
        )

        # Prompt "alpha" imports library "beta", and vice versa
        assert results == ["B", "A"]


class TestReferenceValidation:
    """Dotted access is only a component reference when it names an import."""

    @pytest.mark.asyncio
    async def test_declared_dict_variable_allows_dot_access(self):
        """A declared dict variable is not mistaken for an import alias."""
        assembly = make_assembly(
            ["{{ config.name }}"],
            variables=[{"name": "config", "type": "dict", "description": "d"}],
        )

        result = await PromptCompiler().compile(assembly, {"config": {"name": "pal"}})

        assert result == "pal"

    @pytest.mark.asyncio
    async def test_undeclared_runtime_variable_allows_filtered_dot_access(self):
        """Variables passed at compile time are accepted even with a filter."""
        assembly = make_assembly(["{{ config.name | upper }}"])

        result = await PromptCompiler().compile(assembly, {"config": {"name": "pal"}})

        assert result == "PAL"

    @pytest.mark.asyncio
    async def test_multi_target_loop_variables_are_not_references(self):
        """`{% for k, v in ... %}` binds every name in the target list."""
        assembly = make_assembly(
            [
                "{% for key, entry in items.items() %}",
                "{{ key }}={{ entry.label }}",
                "{% endfor %}",
            ],
            variables=[{"name": "items", "type": "dict", "description": "d"}],
        )

        result = await PromptCompiler().compile(
            assembly, {"items": {"a": {"label": "A"}}}
        )

        assert result == "a=A"

    @pytest.mark.asyncio
    async def test_set_targets_are_not_references(self):
        """`{% set %}` binds a name that is not an import alias."""
        assembly = make_assembly(
            ["{% set alias = source %}{{ alias.key }}"],
            variables=[{"name": "source", "type": "dict", "description": "d"}],
        )

        result = await PromptCompiler().compile(assembly, {"source": {"key": "v"}})

        assert result == "v"

    @pytest.mark.asyncio
    async def test_filtered_component_reference_is_validated(self, temp_dir):
        """A missing component is reported even when a filter follows it."""
        lib = write_library(temp_dir / "p.pal.lib", "p", {"present": "content"})
        assembly = make_assembly(["{{ p.absent | trim }}"], imports={"p": str(lib)})

        with pytest.raises(PALMissingComponentError) as exc_info:
            await PromptCompiler().compile(assembly)

        errors = exc_info.value.context["errors"]
        assert "absent" in errors[0]
        assert "present" in errors[0]

    @pytest.mark.asyncio
    async def test_unknown_alias_is_still_reported(self):
        """A genuinely unknown alias remains an error."""
        assembly = make_assembly(["{{ nowhere.thing }}"])

        with pytest.raises(PALMissingComponentError):
            await PromptCompiler().compile(assembly)

    @pytest.mark.asyncio
    async def test_importing_a_prompt_assembly_explains_the_problem(self, temp_dir):
        """Importing a .pal file names the actual constraint."""
        prompt_file = temp_dir / "other.pal"
        prompt_file.write_text(
            yaml.dump(
                {
                    "pal_version": "1.0",
                    "id": "other",
                    "version": "1.0.0",
                    "description": "d",
                    "composition": ["x"],
                }
            ),
            encoding="utf-8",
        )
        assembly = make_assembly(
            ["{{ other.thing }}"], imports={"other": str(prompt_file)}
        )

        with pytest.raises(PALResolverError) as exc_info:
            await PromptCompiler().compile(assembly)

        assert ".pal.lib" in str(exc_info.value)


class TestFiltersAndErrorWrapping:
    """Filters follow stock Jinja2, and render failures stay inside PALError."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("{{ value | upper }}", "5"),
            ("{{ value | lower }}", "5"),
            ("{{ value | title }}", "5"),
        ],
    )
    @pytest.mark.asyncio
    async def test_case_filters_accept_non_string_values(self, expression, expected):
        """Case filters coerce like Jinja2's built-ins instead of raising."""
        assembly = make_assembly(
            [expression],
            variables=[{"name": "value", "type": "integer", "description": "d"}],
        )

        result = await PromptCompiler().compile(assembly, {"value": 5})

        assert result == expected

    @pytest.mark.asyncio
    async def test_non_template_render_errors_are_wrapped(self):
        """A plain Python error during rendering surfaces as PALCompilerError."""
        assembly = make_assembly(
            ["{{ value + 1 }}"],
            variables=[{"name": "value", "type": "any", "description": "d"}],
        )

        with pytest.raises(PALCompilerError) as exc_info:
            await PromptCompiler().compile(assembly, {"value": "text"})

        assert exc_info.value.context["error_type"] == "TypeError"


class TestVariableDefaults:
    """Declared defaults are treated exactly like supplied values."""

    @pytest.mark.asyncio
    async def test_default_is_type_converted(self):
        """A default is converted to the declared type before rendering."""
        assembly = make_assembly(
            ["{{ count + 1 }}"],
            variables=[
                {
                    "name": "count",
                    "type": "integer",
                    "description": "d",
                    "required": False,
                    "default": "12",
                }
            ],
        )

        assert await PromptCompiler().compile(assembly) == "13"

    @pytest.mark.asyncio
    async def test_invalid_default_raises_a_pal_error(self):
        """A default that cannot be converted is reported, not passed through."""
        assembly = make_assembly(
            ["{{ count }}"],
            variables=[
                {
                    "name": "count",
                    "type": "integer",
                    "description": "d",
                    "required": False,
                    "default": "not-a-number",
                }
            ],
        )

        with pytest.raises(PALCompilerError) as exc_info:
            await PromptCompiler().compile(assembly)

        assert exc_info.value.context["variable"] == "count"

    @pytest.mark.asyncio
    async def test_optional_any_variable_renders_empty(self):
        """An unset optional must not inject the literal "None"."""
        assembly = make_assembly(
            ["Notes: {{ notes }}"],
            variables=[
                {
                    "name": "notes",
                    "type": "any",
                    "description": "d",
                    "required": False,
                }
            ],
        )

        assert await PromptCompiler().compile(assembly) == "Notes:"


class TestPromptCleanup:
    """Whitespace cleanup must not rewrite quoted content."""

    def test_blank_lines_inside_fenced_blocks_are_preserved(self):
        """Code samples keep their internal blank lines."""
        compiler = PromptCompiler()
        prompt = "Intro:\n```python\na = 1\n\n\nb = 2\n```"

        assert compiler._clean_compiled_prompt(prompt) == prompt

    def test_blank_lines_outside_fenced_blocks_are_collapsed(self):
        """Ordinary prose still has excess blank lines removed."""
        compiler = PromptCompiler()

        cleaned = compiler._clean_compiled_prompt("Line 1\n\n\n\nLine 2")

        assert cleaned == "Line 1\n\nLine 2"


class TestCompilerPathHandling:
    """The compiler accepts the same path types as the loader."""

    def test_compile_from_file_accepts_a_string_path(self, temp_dir):
        """A str path resolves imports instead of raising AttributeError."""
        write_library(temp_dir / "p.pal.lib", "p", {"c": "content"})
        pal_file = temp_dir / "prompt.pal"
        pal_file.write_text(
            yaml.dump(
                {
                    "pal_version": "1.0",
                    "id": "str-path",
                    "version": "1.0.0",
                    "description": "d",
                    "imports": {"p": "p.pal.lib"},
                    "composition": ["{{ p.c }}"],
                }
            ),
            encoding="utf-8",
        )

        assert PromptCompiler().compile_from_file_sync(str(pal_file)) == "content"

    def test_analyze_template_variables_reports_root_names(self):
        """Undeclared names are reported by their root, not dotted."""
        assembly = make_assembly(["{{ unknown.thing }} {{ plain }}"])

        found = PromptCompiler().analyze_template_variables(assembly)

        assert found == {"unknown", "plain"}


def _pricing_response(url, payload):
    """Build an httpx response for the pricing endpoint."""
    return httpx.Response(200, text=payload, request=httpx.Request("GET", url))


class TestExecutionCostEstimation:
    """A billed LLM response is never discarded by cost estimation."""

    @pytest.fixture
    def assembly(self):
        """A trivial assembly for execution tests."""
        return make_assembly(["hi"])

    @pytest.mark.asyncio
    async def test_malformed_pricing_data_preserves_the_response(self, assembly):
        """A non-JSON pricing body must not fail an execution."""
        client = MockLLMClient(response="billed answer")
        executor = PromptExecutor(client)

        async def not_json(self, url, **kwargs):
            return _pricing_response(url, "<html>not json</html>")

        with patch.object(httpx.AsyncClient, "get", not_json):
            result = await executor.execute("hi", assembly, "gpt-4")

        assert result.success is True
        assert result.response == "billed answer"
        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_pricing_failures_are_negatively_cached(self, assembly):
        """A broken pricing endpoint is retried per window, not per execution."""
        executor = PromptExecutor(MockLLMClient())
        calls = 0

        async def failing(self, url, **kwargs):
            nonlocal calls
            calls += 1
            raise httpx.RequestError("unreachable")

        with patch.object(httpx.AsyncClient, "get", failing):
            for _ in range(3):
                assert (await executor.execute("hi", assembly, "gpt-4")).success

        assert calls == 1

    @pytest.mark.parametrize(
        "payload",
        [
            '["not", "a", "dict"]',
            '{"gpt-4": "not-a-mapping"}',
            '{"gpt-4": {}}',
            '{"gpt-4": {"input_cost_per_token": "abc", "output_cost_per_token": 1}}',
        ],
    )
    @pytest.mark.asyncio
    async def test_hostile_pricing_payloads_never_raise(self, assembly, payload):
        """Any shape of pricing data degrades to an unknown cost."""
        executor = PromptExecutor(MockLLMClient(response="billed answer"))

        async def hostile(self, url, **kwargs):
            return _pricing_response(url, payload)

        with patch.object(httpx.AsyncClient, "get", hostile):
            result = await executor.execute("hi", assembly, "gpt-4")

        assert result.success is True
        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_valid_pricing_still_produces_a_cost(self, assembly):
        """The happy path still computes a cost from live pricing."""
        executor = PromptExecutor(MockLLMClient())

        async def priced(self, url, **kwargs):
            return _pricing_response(
                url,
                '{"gpt-4": {"input_cost_per_token": 1e-05,'
                ' "output_cost_per_token": 3e-05}}',
            )

        with patch.object(httpx.AsyncClient, "get", priced):
            result = await executor.execute("hi", assembly, "gpt-4")

        assert result.cost_usd is not None
        assert result.cost_usd > 0


class TestExecutionHistoryBounds:
    """Execution history is unbounded by default and capped on request."""

    @pytest.mark.asyncio
    async def test_history_is_unbounded_by_default(self):
        """The default keeps every result, as before."""
        executor = PromptExecutor(MockLLMClient())
        assembly = make_assembly(["hi"])

        for _ in range(4):
            await executor.execute("hi", assembly, "mock")

        assert len(executor.get_execution_history()) == 4

    @pytest.mark.asyncio
    async def test_max_history_keeps_the_most_recent_results(self):
        """A configured cap discards the oldest results."""
        executor = PromptExecutor(MockLLMClient(), max_history=2)
        assembly = make_assembly(["hi"])

        for _ in range(4):
            await executor.execute("hi", assembly, "mock")

        history = executor.get_execution_history()
        assert len(history) == 2


class TestLoaderLifecycleAndLimits:
    """The loader can be closed and refuses oversized documents."""

    @pytest.mark.asyncio
    async def test_aclose_closes_a_lazily_created_client(self):
        """A client created outside the context manager can still be closed."""
        loader = Loader()
        loader._http_client = httpx.AsyncClient(timeout=loader.timeout)

        await loader.aclose()

        assert loader._http_client.is_closed

    @pytest.mark.asyncio
    async def test_oversized_download_is_rejected(self):
        """A response beyond the limit raises PALLoadError."""
        loader = Loader(max_download_bytes=16)
        url = "https://example.com/big.pal"

        async def oversized(self, request_url, **kwargs):
            return httpx.Response(
                200,
                content=b"x" * 64,
                request=httpx.Request("GET", request_url),
            )

        with (
            patch.object(httpx.AsyncClient, "get", oversized),
            pytest.raises(PALLoadError) as exc_info,
        ):
            await loader.load_prompt_assembly_async(url)

        assert exc_info.value.context["limit"] == 16

    @pytest.mark.asyncio
    async def test_limit_can_be_disabled(self):
        """Passing None removes the ceiling."""
        loader = Loader(max_download_bytes=None)
        body = yaml.dump(
            {
                "pal_version": "1.0",
                "id": "big",
                "version": "1.0.0",
                "description": "d",
                "composition": ["x"],
            }
        )

        async def large(self, request_url, **kwargs):
            return httpx.Response(
                200, text=body, request=httpx.Request("GET", request_url)
            )

        with patch.object(httpx.AsyncClient, "get", large):
            assembly = await loader.load_prompt_assembly_async(
                "https://example.com/big.pal"
            )

        assert assembly.id == "big"


class TestEvaluationImportResolution:
    """Evaluations resolve imports against the prompt file, not the CWD."""

    @pytest.fixture
    def project(self, temp_dir):
        """A prompt importing a library from a sibling directory."""
        (temp_dir / "libs").mkdir()
        (temp_dir / "prompts").mkdir()
        write_library(
            temp_dir / "libs" / "p.pal.lib", "p", {"expert": "You are an expert."}
        )
        (temp_dir / "prompts" / "t.pal").write_text(
            yaml.dump(
                {
                    "pal_version": "1.0",
                    "id": "eval-target",
                    "version": "1.0.0",
                    "description": "d",
                    "imports": {"p": "../libs/p.pal.lib"},
                    "composition": ["{{ p.expert }}"],
                }
            ),
            encoding="utf-8",
        )
        (temp_dir / "prompts" / "t.eval.yaml").write_text(
            yaml.dump(
                {
                    "pal_version": "1.0",
                    "prompt_id": "eval-target",
                    "target_version": "1.0.0",
                    "test_cases": [
                        {
                            "name": "case",
                            "variables": {},
                            "assertions": [
                                {"type": "contains", "config": {"text": "Mock"}}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return temp_dir

    @pytest.mark.parametrize("explicit_pal_file", [True, False])
    @pytest.mark.asyncio
    async def test_relative_imports_resolve_from_an_unrelated_cwd(
        self, project, monkeypatch, tmp_path, explicit_pal_file
    ):
        """Both the explicit and discovered prompt paths compile correctly."""
        monkeypatch.chdir(tmp_path)

        loader = Loader()
        runner = EvaluationRunner(
            loader, PromptCompiler(loader), PromptExecutor(MockLLMClient())
        )

        result = await runner.run_evaluation(
            project / "prompts" / "t.eval.yaml",
            (project / "prompts" / "t.pal") if explicit_pal_file else None,
            model="mock",
        )

        test_result = result.test_results[0]
        assert test_result.error is None
        assert test_result.execution_result.compiled_prompt == "You are an expert."


class TestAssertionConfigValidation:
    """A malformed assertion config fails that assertion, not the test case."""

    def test_regex_flags_accept_names(self):
        """Flag names are usable from YAML, where re constants are not."""
        result = RegexMatchAssertion().evaluate(
            "HELLO",
            {
                "pattern": "hello",
                "flags": "IGNORECASE",
            },
        )

        assert result.passed is True

    def test_unknown_regex_flag_fails_the_assertion(self):
        """An unrecognised flag name is reported without raising."""
        result = RegexMatchAssertion().evaluate(
            "hello",
            {
                "pattern": "hello",
                "flags": "NOPE",
            },
        )

        assert result.passed is False
        assert "NOPE" in result.message

    def test_string_length_bounds_are_coerced(self):
        """Numeric strings from YAML are accepted as bounds."""
        result = LengthAssertion().evaluate("hello", {"min_length": "3"})

        assert result.passed is True

    def test_non_numeric_length_bound_fails_the_assertion(self):
        """A bound that is not a number is reported without raising."""
        result = LengthAssertion().evaluate("hello", {"min_length": "three"})

        assert result.passed is False
        assert "min_length" in result.message

    def test_non_string_contains_text_fails_the_assertion(self):
        """A non-string 'text' is reported without raising."""
        result = ContainsAssertion().evaluate("hello", {"text": ["hello"]})

        assert result.passed is False
        assert "must be a string" in result.message
