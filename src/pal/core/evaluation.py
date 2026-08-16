"""PAL evaluation system for testing prompt assemblies."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog

from ..exceptions.core import PALError
from ..models.schema import (
    EvaluationSuite,
    EvaluationTestCase,
    ExecutionResult,
    PromptAssembly,
)
from .compiler import PromptCompiler
from .executor import PromptExecutor
from .loader import Loader

logger = structlog.get_logger()


class AssertionResult:
    """Result of an evaluation assertion."""

    def __init__(
        self,
        assertion_type: str,
        passed: bool,
        message: str,
        expected: Any = None,
        actual: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.assertion_type = assertion_type
        self.passed = passed
        self.message = message
        self.expected = expected
        self.actual = actual
        self.metadata = metadata or {}


class TestCaseResult:
    """Result of an evaluation test case."""

    def __init__(
        self,
        test_case: EvaluationTestCase,
        execution_result: ExecutionResult | None = None,
        assertion_results: list[AssertionResult] | None = None,
        error: str | None = None,
    ) -> None:
        self.test_case = test_case
        self.execution_result = execution_result
        self.assertion_results = assertion_results or []
        self.error = error
        self.passed = error is None and all(ar.passed for ar in self.assertion_results)


class EvaluationResult:
    """Result of an evaluation suite."""

    def __init__(
        self, evaluation_suite: EvaluationSuite, test_results: list[TestCaseResult]
    ) -> None:
        self.evaluation_suite = evaluation_suite
        self.test_results = test_results
        self.total_tests = len(test_results)
        self.passed_tests = sum(1 for tr in test_results if tr.passed)
        self.failed_tests = self.total_tests - self.passed_tests
        self.pass_rate = (
            self.passed_tests / self.total_tests if self.total_tests > 0 else 0
        )


class BaseAssertion(ABC):
    """Base class for evaluation assertions."""

    @abstractmethod
    def evaluate(self, response: str, config: dict[str, Any]) -> AssertionResult:
        """Evaluate the assertion against a response."""


class ContainsAssertion(BaseAssertion):
    """Assert that response contains specific text."""

    def evaluate(self, response: str, config: dict[str, Any]) -> AssertionResult:
        text = config.get("text", "")
        case_sensitive = bool(config.get("case_sensitive", True))

        if not text:
            return AssertionResult(
                "contains", False, "Missing 'text' parameter in assertion config"
            )

        if not isinstance(text, str):
            return AssertionResult(
                "contains",
                False,
                f"'text' must be a string, got {type(text).__name__}",
            )

        search_text = text if case_sensitive else text.lower()
        search_response = response if case_sensitive else response.lower()

        passed = search_text in search_response
        message = f"Response {'contains' if passed else 'does not contain'} '{text}'"

        return AssertionResult(
            "contains",
            passed,
            message,
            expected=text,
            actual=response[:100] + "..." if len(response) > 100 else response,
        )


class RegexMatchAssertion(BaseAssertion):
    """Assert that response matches a regex pattern."""

    #: Flag names accepted in place of an integer value
    _FLAG_NAMES = {
        "IGNORECASE": re.IGNORECASE,
        "I": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "M": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "S": re.DOTALL,
        "VERBOSE": re.VERBOSE,
        "X": re.VERBOSE,
        "ASCII": re.ASCII,
        "A": re.ASCII,
    }

    def evaluate(self, response: str, config: dict[str, Any]) -> AssertionResult:
        pattern = config.get("pattern", "")

        if not pattern:
            return AssertionResult(
                "regex_match", False, "Missing 'pattern' parameter in assertion config"
            )

        try:
            flags = self._parse_flags(config.get("flags", 0))
        except ValueError as e:
            return AssertionResult("regex_match", False, f"Invalid 'flags' value: {e}")

        try:
            regex = re.compile(pattern, flags)
            match = regex.search(response)
            passed = match is not None

            message = f"Response {'matches' if passed else 'does not match'} pattern '{pattern}'"
            metadata = {"match": match.group(0) if match else None}

            return AssertionResult(
                "regex_match",
                passed,
                message,
                expected=pattern,
                actual=response[:100] + "..." if len(response) > 100 else response,
                metadata=metadata,
            )
        except re.error as e:
            return AssertionResult(
                "regex_match", False, f"Invalid regex pattern '{pattern}': {e}"
            )

    @classmethod
    def _parse_flags(cls, flags: Any) -> int:
        """Accept an int, a flag name, or a list/'|'-separated set of names."""
        if isinstance(flags, bool):
            raise ValueError(f"expected an int or flag name, got {flags!r}")
        if isinstance(flags, int):
            return flags

        if isinstance(flags, str):
            names = [part.strip() for part in flags.replace("|", ",").split(",")]
        elif isinstance(flags, list):
            names = [str(part).strip() for part in flags]
        else:
            raise ValueError(
                f"expected an int or flag name, got {type(flags).__name__}"
            )

        value = 0
        for name in names:
            if not name:
                continue
            try:
                value |= cls._FLAG_NAMES[name.upper()]
            except KeyError:
                raise ValueError(f"unknown regex flag '{name}'") from None
        return value


class JSONValidAssertion(BaseAssertion):
    """Assert that response is valid JSON."""

    def evaluate(self, response: str, config: dict[str, Any]) -> AssertionResult:
        try:
            parsed = json.loads(response.strip())
            return AssertionResult(
                "json_valid",
                True,
                "Response is valid JSON",
                expected="valid JSON",
                actual="valid JSON",
                metadata={"parsed": parsed},
            )
        except json.JSONDecodeError as e:
            return AssertionResult(
                "json_valid",
                False,
                f"Response is not valid JSON: {e}",
                expected="valid JSON",
                actual=response[:100] + "..." if len(response) > 100 else response,
            )


class JSONFieldEqualsAssertion(BaseAssertion):
    """Assert that a JSON field equals a specific value."""

    def evaluate(self, response: str, config: dict[str, Any]) -> AssertionResult:
        path = config.get("path", "")
        expected_value = config.get("value")

        if not path:
            return AssertionResult(
                "json_field_equals",
                False,
                "Missing 'path' parameter in assertion config",
            )

        try:
            parsed = json.loads(response.strip())
            actual_value = self._extract_json_path(parsed, path)

            passed = actual_value == expected_value
            message = f"JSON field '{path}' {'equals' if passed else 'does not equal'} expected value"

            return AssertionResult(
                "json_field_equals",
                passed,
                message,
                expected=expected_value,
                actual=actual_value,
            )
        except json.JSONDecodeError as e:
            return AssertionResult(
                "json_field_equals", False, f"Response is not valid JSON: {e}"
            )
        except (KeyError, TypeError, IndexError) as e:
            return AssertionResult(
                "json_field_equals",
                False,
                f"Cannot extract path '{path}' from JSON: {e}",
            )

    def _extract_json_path(self, data: Any, path: str) -> Any:
        """Extract value from JSON using JSONPath-like syntax."""
        # Simple implementation for basic paths like $.field or $.field[0]
        if path.startswith("$."):
            path = path[2:]  # Remove $.

        parts = path.replace("[", ".").replace("]", "").split(".")
        current = data

        for part in parts:
            current = current[int(part)] if part.isdigit() else current[part]

        return current


class LengthAssertion(BaseAssertion):
    """Assert response length constraints."""

    def evaluate(self, response: str, config: dict[str, Any]) -> AssertionResult:
        try:
            min_length = self._as_int(config, "min_length")
            max_length = self._as_int(config, "max_length")
            exact_length = self._as_int(config, "exact_length")
        except ValueError as e:
            return AssertionResult("length", False, str(e))

        response_length = len(response)

        if exact_length is not None:
            passed = response_length == exact_length
            message = (
                f"Response length is {response_length}, expected exactly {exact_length}"
            )
        elif min_length is not None and max_length is not None:
            passed = min_length <= response_length <= max_length
            message = f"Response length is {response_length}, expected between {min_length} and {max_length}"
        elif min_length is not None:
            passed = response_length >= min_length
            message = (
                f"Response length is {response_length}, expected at least {min_length}"
            )
        elif max_length is not None:
            passed = response_length <= max_length
            message = (
                f"Response length is {response_length}, expected at most {max_length}"
            )
        else:
            return AssertionResult(
                "length", False, "No length constraints specified in assertion config"
            )

        return AssertionResult(
            "length", passed, message, expected=config, actual=response_length
        )

    @staticmethod
    def _as_int(config: dict[str, Any], key: str) -> int | None:
        """Read an optional integer constraint from the config."""
        value = config.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | str | float):
            raise ValueError(f"'{key}' must be an integer, got {type(value).__name__}")
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"'{key}' must be an integer, got {value!r}") from e


class EvaluationRunner:
    """Runs evaluation suites against prompt assemblies."""

    def __init__(
        self, loader: Loader, compiler: PromptCompiler, executor: PromptExecutor
    ) -> None:
        self.loader = loader
        self.compiler = compiler
        self.executor = executor

        # Register built-in assertions
        self.assertions: dict[str, BaseAssertion] = {
            "contains": ContainsAssertion(),
            "regex_match": RegexMatchAssertion(),
            "json_valid": JSONValidAssertion(),
            "json_field_equals": JSONFieldEqualsAssertion(),
            "length": LengthAssertion(),
        }

    def register_assertion(self, name: str, assertion: BaseAssertion) -> None:
        """Register a custom assertion."""
        self.assertions[name] = assertion

    async def run_evaluation(
        self,
        eval_file: Path,
        pal_file: Path | None = None,
        model: str = "mock",
        **execution_kwargs: Any,
    ) -> EvaluationResult:
        """Run an evaluation suite."""
        # Load evaluation suite
        evaluation_suite = await self.loader.load_evaluation_suite_async(eval_file)

        # Load prompt assembly. The originating path is kept so that relative
        # imports resolve against the .pal file rather than the process CWD.
        if pal_file:
            prompt_assembly = await self.loader.load_prompt_assembly_async(pal_file)
            prompt_path: Path | None = Path(pal_file)
        else:
            # Find PAL file by prompt_id
            prompt_assembly, prompt_path = await self._find_prompt_assembly(
                evaluation_suite.prompt_id, Path(eval_file).parent
            )

        # Validate version compatibility
        if prompt_assembly.version != evaluation_suite.target_version:
            logger.warning(
                "Version mismatch",
                prompt_version=prompt_assembly.version,
                target_version=evaluation_suite.target_version,
            )

        # Run test cases
        test_results: list[TestCaseResult] = []

        for test_case in evaluation_suite.test_cases:
            test_result = await self._run_test_case(
                test_case, prompt_assembly, model, prompt_path, **execution_kwargs
            )
            test_results.append(test_result)

        return EvaluationResult(evaluation_suite, test_results)

    async def _run_test_case(
        self,
        test_case: EvaluationTestCase,
        prompt_assembly: PromptAssembly,
        model: str,
        base_path: Path | None = None,
        **execution_kwargs: Any,
    ) -> TestCaseResult:
        """Run a single test case."""
        try:
            # Compile prompt with test variables
            compiled_prompt = await self.compiler.compile(
                prompt_assembly, test_case.variables, base_path
            )

            # Execute prompt
            execution_result = await self.executor.execute(
                compiled_prompt, prompt_assembly, model, **execution_kwargs
            )

            # Run assertions
            assertion_results = []
            for assertion_def in test_case.assertions:
                assertion = self.assertions.get(assertion_def.type)
                if not assertion:
                    assertion_results.append(
                        AssertionResult(
                            assertion_def.type,
                            False,
                            f"Unknown assertion type: {assertion_def.type}",
                        )
                    )
                    continue

                result = assertion.evaluate(
                    execution_result.response, assertion_def.config
                )
                assertion_results.append(result)

            return TestCaseResult(test_case, execution_result, assertion_results)

        except Exception as e:
            logger.error(
                "Test case execution failed", test_case=test_case.name, error=str(e)
            )
            return TestCaseResult(test_case, error=str(e))

    async def _find_prompt_assembly(
        self, prompt_id: str, search_dir: Path
    ) -> tuple[PromptAssembly, Path]:
        """Find prompt assembly file by ID.

        Returns:
            The matching assembly and the path it was loaded from
        """
        # Look for .pal files in the directory
        for pal_file in search_dir.glob("**/*.pal"):
            try:
                assembly = await self.loader.load_prompt_assembly_async(pal_file)
                if assembly.id == prompt_id:
                    return assembly, pal_file
            except Exception:
                continue  # Skip invalid files

        raise PALError(
            f"Could not find prompt assembly with ID '{prompt_id}' in {search_dir}"
        )


class EvaluationReporter:
    """Generates reports from evaluation results."""

    def generate_console_report(self, result: EvaluationResult) -> str:
        """Generate a console-friendly report."""
        lines = []
        lines.append(f"Evaluation Report: {result.evaluation_suite.prompt_id}")
        lines.append("=" * 50)
        lines.append(f"Total Tests: {result.total_tests}")
        lines.append(f"Passed: {result.passed_tests}")
        lines.append(f"Failed: {result.failed_tests}")
        lines.append(f"Pass Rate: {result.pass_rate:.1%}")
        lines.append("")

        for test_result in result.test_results:
            status = "✓" if test_result.passed else "✗"
            lines.append(f"{status} {test_result.test_case.name}")

            if test_result.error:
                lines.append(f"   Error: {test_result.error}")
            else:
                for assertion_result in test_result.assertion_results:
                    assertion_status = "✓" if assertion_result.passed else "✗"
                    lines.append(f"   {assertion_status} {assertion_result.message}")

            lines.append("")

        return "\n".join(lines)

    def generate_json_report(self, result: EvaluationResult) -> dict[str, Any]:
        """Generate a JSON report."""
        return {
            "evaluation_suite": result.evaluation_suite.model_dump(),
            "summary": {
                "total_tests": result.total_tests,
                "passed_tests": result.passed_tests,
                "failed_tests": result.failed_tests,
                "pass_rate": result.pass_rate,
            },
            "test_results": [
                {
                    "test_case": tr.test_case.model_dump(),
                    "passed": tr.passed,
                    "error": tr.error,
                    "execution_result": tr.execution_result.model_dump()
                    if tr.execution_result
                    else None,
                    "assertion_results": [
                        {
                            "type": ar.assertion_type,
                            "passed": ar.passed,
                            "message": ar.message,
                            "expected": ar.expected,
                            "actual": ar.actual,
                            "metadata": ar.metadata,
                        }
                        for ar in tr.assertion_results
                    ],
                }
                for tr in result.test_results
            ],
        }
