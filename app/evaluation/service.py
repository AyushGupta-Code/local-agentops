"""Evaluation harness for first-pass retrieval and generation benchmarking."""

import json
from pathlib import Path
from time import perf_counter

from app.core.config import Settings, get_settings
from app.evaluation.validation import run_default_validations
from app.generation.service import GenerationService
from app.models.schemas import EvaluationRunRecord, EvaluationTestCase, GenerationRequest


class EvaluationService:
    """Service for running auditable benchmark cases against the local stack."""

    def __init__(
        self,
        *,
        generation_service: GenerationService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Store the generation dependency and managed evaluation paths.

        Why this function exists:
            The evaluation harness should remain a thin coordinator rather than
            embedding generation or storage logic directly. Injecting the
            generation service keeps the harness modular and allows tests to use
            mocked local backends while production code can reuse the real
            generation pipeline.

        Parameters:
            generation_service: Optional generation service used to execute
                benchmark cases.
            settings: Optional validated settings object used for saving local
                evaluation artifacts.

        Returns:
            This initializer stores the dependencies on the service instance.

        Edge cases handled:
            A default generation service and settings object are created
            automatically so the benchmark runner script can stay compact.
        """
        self._settings = settings or get_settings()
        self._generation_service = generation_service or GenerationService(settings=self._settings)

    def run_case(
        self,
        *,
        dataset_name: str,
        test_case: EvaluationTestCase,
    ) -> EvaluationRunRecord:
        """Run one benchmark case and record its generation and validation results.

        Why this function exists:
            A benchmark harness should produce a single auditable record for
            each case. This method builds the generation request, measures
            latency, executes the generation path, runs deterministic
            validations, and returns a structured record that can later be saved
            or inspected.

        Parameters:
            dataset_name: Name of the benchmark collection the case belongs to.
            test_case: Deterministic evaluation case to run.

        Returns:
            An ``EvaluationRunRecord`` describing the generation result and all
            first-pass validation outcomes.

        Edge cases handled:
            Generation failures are captured as failed records with explicit
            notes and zero citations rather than crashing the entire benchmark
            run.
        """
        generation_request = GenerationRequest(
            query=test_case.query,
            task_type=test_case.task_type,
            retrieved_context=test_case.retrieved_context,
            output_instructions=test_case.output_instructions,
        )

        start_time = perf_counter()
        try:
            response = self._generation_service.answer(generation_request)
            generation_error = None
        except Exception as exc:  # pragma: no cover - exercised by behavior, not exception type
            response = None
            generation_error = str(exc)
        latency_ms = (perf_counter() - start_time) * 1000.0

        if response is None:
            validation_outcomes = []
            notes = [f"Generation failed: {generation_error}"]
            return EvaluationRunRecord(
                evaluation_id=test_case.evaluation_id,
                dataset_name=dataset_name,
                query=test_case.query,
                task_type=test_case.task_type,
                retrieved_chunk_ids=[result.chunk_id for result in test_case.retrieved_context],
                selected_evidence_chunk_ids=[],
                final_answer=None,
                citations=[],
                supporting_evidence_summary=[],
                latency_ms=latency_ms,
                validation_outcomes=validation_outcomes,
                passed=False,
                notes=notes,
            )

        validation_outcomes = run_default_validations(
            response=response,
            retrieved_context=test_case.retrieved_context,
        )
        passed = all(
            outcome.passed
            for outcome in validation_outcomes
            if outcome.severity == "error"
        )

        return EvaluationRunRecord(
            evaluation_id=test_case.evaluation_id,
            dataset_name=dataset_name,
            query=test_case.query,
            task_type=test_case.task_type,
            retrieved_chunk_ids=[result.chunk_id for result in test_case.retrieved_context],
            selected_evidence_chunk_ids=list(response.used_chunk_ids),
            final_answer=response.answer,
            citations=list(response.citations),
            supporting_evidence_summary=list(response.supporting_evidence_summary),
            latency_ms=latency_ms,
            validation_outcomes=validation_outcomes,
            passed=passed,
            notes=[],
        )

    def run_benchmark(
        self,
        *,
        dataset_name: str,
        test_cases: list[EvaluationTestCase],
        save_results: bool = True,
    ) -> list[EvaluationRunRecord]:
        """Run a deterministic benchmark suite and optionally persist the results.

        Why this function exists:
            Benchmarking usually happens over multiple fixed test cases. This
            method keeps that workflow understandable by executing cases
            sequentially, collecting per-case records, and optionally writing a
            single auditable JSON report under the managed evaluation
            directory.

        Parameters:
            dataset_name: Name of the benchmark collection being executed.
            test_cases: Ordered list of evaluation cases to run.
            save_results: Whether the aggregated results should be saved to a
                local file.

        Returns:
            A list of ``EvaluationRunRecord`` objects in execution order.

        Edge cases handled:
            Empty benchmark suites return an empty list cleanly, and result
            saving can be disabled for callers that only need in-memory records.
        """
        records = [
            self.run_case(dataset_name=dataset_name, test_case=test_case)
            for test_case in test_cases
        ]

        if save_results:
            self.save_results(dataset_name=dataset_name, records=records)

        return records

    def save_results(
        self,
        *,
        dataset_name: str,
        records: list[EvaluationRunRecord],
    ) -> Path:
        """Persist aggregated benchmark results to a managed local JSON file.

        Why this function exists:
            Evaluation runs should leave behind inspectable artifacts so later
            changes can be audited and compared. This method writes a compact
            JSON report under the managed evaluation directory rather than
            scattering ad hoc output files across the repository.

        Parameters:
            dataset_name: Benchmark collection name used for the output path.
            records: Evaluation run records to serialize.

        Returns:
            The path of the written JSON report.

        Edge cases handled:
            Parent directories are created automatically, and even an empty
            record list produces a valid report file for explicit no-op runs.
        """
        output_path = self._settings.resolve_eval_data_path(f"{dataset_name}/latest_results.json")
        payload = {
            "dataset_name": dataset_name,
            "case_count": len(records),
            "records": [record.model_dump(mode="json") for record in records],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
