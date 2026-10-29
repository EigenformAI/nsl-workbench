import unittest

from src.typing.trajectory import EpisodeTrajectory, GenerationData


class TrajectoryTypeTests(unittest.TestCase):
    def make_episode(
        self,
        episode_index: int,
        *,
        row_count: int = 1,
        success: bool = True,
        episode_runtime_success: bool | None = None,
        score: float = 128.0,
        partial: bool = False,
        error_message: str | None = None,
        duration_seconds: float = 3.0,
        container_overhead_seconds: float | None = None,
        episode_execution_seconds: float | None = None,
        total_inference_ms: float | None = None,
        inference_call_count: int | None = None,
        average_output_tokens_per_second: float | None = None,
        inference_duty_cycle: float | None = None,
        peak_gpu_utilization_pct: float | None = None,
        peak_cpu_utilization_pct: float | None = None,
        avg_gpu_utilization_pct: float | None = None,
        avg_cpu_utilization_pct: float | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        peak_context_tokens: int | None = None,
        avg_context_tokens: float | None = None,
        median_context_tokens: float | None = None,
        error_category: str | None = None,
        peak_kv_cache_usage_pct: float | None = None,
        avg_kv_cache_usage_pct: float | None = None,
        peak_num_requests_running: int | None = None,
        peak_num_requests_waiting: int | None = None,
        tool_calls_count: int = 0,
    ) -> EpisodeTrajectory:
        prompt_responses = [
            {
                "prompt": f"prompt-{episode_index}-{idx}",
                "raw_response": f"response-{episode_index}-{idx}",
                "interaction_type": "explorer",
                "timestamp": "2026-04-06T00:00:00",
                "success": True,
                "error_message": None,
            }
            for idx in range(row_count)
        ]
        return EpisodeTrajectory(
            episode_id=f"ep-{episode_index}",
            generation_id=7,
            episode_index=episode_index,
            prompt_responses=prompt_responses,
            trajectory={"mode_history": []},
            score=score,
            episode_runtime_success=(
                episode_runtime_success
                if episode_runtime_success is not None
                else score > 0.0
            ),
            success=success,
            llm_turns_count=3,
            tool_calls_count=tool_calls_count,
            container_variation="variation_1_heavy",
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T00:00:03",
            duration_seconds=duration_seconds,
            partial=partial,
            error_message=error_message,
            error_category=error_category,
            task_breakdown={
                "space_measurements": {"container-a": (1024.0, 1152.0)},
                "filesystem_groups": [
                    {
                        "filesystem_id": "fs-shared",
                        "container_ids": ["container-a", "container-b"],
                    }
                ],
                **({"measurement_errors": ["container-b"]} if error_message else {}),
            },
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=episode_execution_seconds,
            total_inference_ms=total_inference_ms,
            inference_call_count=inference_call_count,
            average_output_tokens_per_second=average_output_tokens_per_second,
            inference_duty_cycle=inference_duty_cycle,
            peak_gpu_utilization_pct=peak_gpu_utilization_pct,
            peak_cpu_utilization_pct=peak_cpu_utilization_pct,
            avg_gpu_utilization_pct=avg_gpu_utilization_pct,
            avg_cpu_utilization_pct=avg_cpu_utilization_pct,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            peak_context_tokens=peak_context_tokens,
            avg_context_tokens=avg_context_tokens,
            median_context_tokens=median_context_tokens,
            peak_kv_cache_usage_pct=peak_kv_cache_usage_pct,
            avg_kv_cache_usage_pct=avg_kv_cache_usage_pct,
            peak_num_requests_running=peak_num_requests_running,
            peak_num_requests_waiting=peak_num_requests_waiting,
        )

    def test_generation_data_episode_accounting_contract(self) -> None:
        generation_data = GenerationData(generation_id=7)
        generation_data.add_episode(self.make_episode(0, row_count=2, success=True))
        generation_data.add_episode(self.make_episode(1, row_count=3, success=True))
        generation_data.add_episode(self.make_episode(2, row_count=4, success=False))

        self.assertEqual(generation_data.total_episodes_run, 3)
        self.assertEqual(generation_data.total_successful, 2)
        self.assertEqual(generation_data.training_row_count, 5)
        self.assertEqual(len(generation_data.successful_episodes), 2)
        self.assertEqual(len(generation_data.failed_episodes), 1)
        self.assertEqual(generation_data.success_rate, 2 / 3)

    def test_episode_trajectory_core_serialization_contract(self) -> None:
        episode = self.make_episode(
            6,
            row_count=2,
            success=False,
            episode_runtime_success=False,
            partial=True,
            error_message="measurement failed",
            tool_calls_count=2,
        )

        payload = episode.to_dict()

        self.assertEqual(payload["episode_id"], "ep-6")
        self.assertEqual(payload["generation_id"], 7)
        self.assertEqual(payload["episode_index"], 6)
        self.assertEqual(len(payload["prompt_responses"]), 2)
        self.assertEqual(payload["duration_seconds"], 3.0)
        self.assertEqual(payload["tool_calls_count"], 2)
        self.assertEqual(payload["container_variation"], "variation_1_heavy")
        self.assertEqual(
            payload["task_breakdown"]["space_measurements"],
            {"container-a": (1024.0, 1152.0)},
        )
        self.assertEqual(
            payload["task_breakdown"]["filesystem_groups"],
            [
                {
                    "filesystem_id": "fs-shared",
                    "container_ids": ["container-a", "container-b"],
                }
            ],
        )
        self.assertFalse(payload["episode_runtime_success"])
        self.assertTrue(payload["partial"])
        self.assertEqual(payload["error_message"], "measurement failed")
        self.assertEqual(
            payload["task_breakdown"]["measurement_errors"], ["container-b"]
        )

    def test_episode_trajectory_serialization_includes_observability_fields(
        self,
    ) -> None:
        episode = self.make_episode(
            7,
            success=True,
            container_overhead_seconds=1.25,
            episode_execution_seconds=8.5,
            total_inference_ms=6200.0,
            inference_call_count=4,
            average_output_tokens_per_second=123.4,
            inference_duty_cycle=0.64,
            peak_gpu_utilization_pct=87.0,
            peak_cpu_utilization_pct=42.0,
            avg_gpu_utilization_pct=72.0,
            avg_cpu_utilization_pct=35.0,
            total_input_tokens=250,
            total_output_tokens=80,
            peak_context_tokens=120,
            avg_context_tokens=100.0,
            median_context_tokens=98.0,
            error_category="context_overflow",
            peak_kv_cache_usage_pct=85.0,
            avg_kv_cache_usage_pct=73.0,
            peak_num_requests_running=7,
            peak_num_requests_waiting=3,
        )

        payload = episode.to_dict()
        restored = EpisodeTrajectory.from_dict(payload)

        self.assertEqual(payload["container_overhead_seconds"], 1.25)
        self.assertEqual(payload["episode_execution_seconds"], 8.5)
        self.assertEqual(payload["total_inference_ms"], 6200.0)
        self.assertEqual(payload["inference_call_count"], 4)
        self.assertEqual(payload["average_output_tokens_per_second"], 123.4)
        self.assertEqual(payload["inference_duty_cycle"], 0.64)
        self.assertEqual(payload["peak_gpu_utilization_pct"], 87.0)
        self.assertEqual(payload["peak_cpu_utilization_pct"], 42.0)
        self.assertEqual(payload["avg_gpu_utilization_pct"], 72.0)
        self.assertEqual(payload["avg_cpu_utilization_pct"], 35.0)
        self.assertEqual(payload["total_input_tokens"], 250)
        self.assertEqual(payload["total_output_tokens"], 80)
        self.assertEqual(payload["peak_context_tokens"], 120)
        self.assertEqual(payload["avg_context_tokens"], 100.0)
        self.assertEqual(payload["median_context_tokens"], 98.0)
        self.assertEqual(payload["error_category"], "context_overflow")
        self.assertEqual(payload["peak_kv_cache_usage_pct"], 85.0)
        self.assertEqual(payload["avg_kv_cache_usage_pct"], 73.0)
        self.assertEqual(payload["peak_num_requests_running"], 7)
        self.assertEqual(payload["peak_num_requests_waiting"], 3)
        self.assertEqual(restored.average_output_tokens_per_second, 123.4)
        self.assertEqual(restored.peak_gpu_utilization_pct, 87.0)
        self.assertEqual(restored.avg_gpu_utilization_pct, 72.0)
        self.assertEqual(restored.total_input_tokens, 250)
        self.assertEqual(restored.total_output_tokens, 80)
        self.assertEqual(restored.peak_context_tokens, 120)
        self.assertEqual(restored.avg_context_tokens, 100.0)
        self.assertEqual(restored.median_context_tokens, 98.0)
        self.assertEqual(restored.error_category, "context_overflow")
        self.assertEqual(restored.peak_kv_cache_usage_pct, 85.0)
        self.assertEqual(restored.avg_kv_cache_usage_pct, 73.0)
        self.assertEqual(restored.peak_num_requests_running, 7)
        self.assertEqual(restored.peak_num_requests_waiting, 3)

    def test_generation_metadata_includes_throughput_and_utilization_summary(
        self,
    ) -> None:
        generation_data = GenerationData(
            generation_id=7,
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T01:00:00",
        )
        generation_data.add_episode(
            self.make_episode(
                0,
                row_count=2,
                success=True,
                average_output_tokens_per_second=100.0,
                inference_duty_cycle=0.5,
                peak_gpu_utilization_pct=70.0,
                peak_cpu_utilization_pct=30.0,
                avg_gpu_utilization_pct=55.0,
                avg_cpu_utilization_pct=22.0,
                container_overhead_seconds=2.0,
                episode_execution_seconds=8.0,
                total_inference_ms=5000.0,
            )
        )
        generation_data.add_episode(
            self.make_episode(
                1,
                row_count=4,
                success=True,
                average_output_tokens_per_second=200.0,
                inference_duty_cycle=0.75,
                peak_gpu_utilization_pct=90.0,
                peak_cpu_utilization_pct=45.0,
                avg_gpu_utilization_pct=75.0,
                avg_cpu_utilization_pct=38.0,
                container_overhead_seconds=1.0,
                episode_execution_seconds=9.0,
                total_inference_ms=7500.0,
            )
        )

        metadata = generation_data.to_metadata_dict(run_id="run-123")

        self.assertEqual(metadata["episodes_per_hour"], 2.0)
        self.assertEqual(metadata["training_rows_per_hour"], 6.0)
        self.assertEqual(metadata["average_output_tokens_per_second"], 150.0)
        self.assertEqual(metadata["average_inference_duty_cycle"], 0.625)
        self.assertEqual(metadata["peak_gpu_utilization_pct"], 90.0)
        self.assertEqual(metadata["peak_cpu_utilization_pct"], 45.0)
        self.assertEqual(metadata["average_gpu_utilization_pct"], 65.0)
        self.assertEqual(metadata["average_cpu_utilization_pct"], 30.0)
        self.assertEqual(metadata["total_container_overhead_seconds"], 3.0)
        self.assertEqual(metadata["total_inference_seconds"], 12.5)
        self.assertEqual(metadata["generation_wall_clock_seconds"], 3600.0)

    def test_generation_metadata_includes_termination_reason_and_vllm_fields(
        self,
    ) -> None:
        generation_data = GenerationData(
            generation_id=7,
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T00:30:00",
            termination_reason="deadline_exceeded",
        )
        generation_data.add_episode(
            self.make_episode(
                0,
                row_count=2,
                success=False,
                peak_kv_cache_usage_pct=71.0,
                avg_kv_cache_usage_pct=63.0,
                peak_num_requests_running=4,
                peak_num_requests_waiting=1,
            )
        )
        generation_data.add_episode(
            self.make_episode(
                1,
                row_count=0,
                success=False,
                peak_kv_cache_usage_pct=88.0,
                avg_kv_cache_usage_pct=77.0,
                peak_num_requests_running=6,
                peak_num_requests_waiting=3,
            )
        )

        metadata = generation_data.to_metadata_dict(run_id="run-123")

        self.assertEqual(metadata["termination_reason"], "deadline_exceeded")
        self.assertEqual(metadata["peak_kv_cache_usage_pct"], 88.0)
        self.assertEqual(metadata["avg_kv_cache_usage_pct"], 70.0)
        self.assertEqual(metadata["peak_num_requests_running"], 6)
        self.assertEqual(metadata["peak_num_requests_waiting"], 3)

    def test_generation_metadata_includes_enriched_metrics(self) -> None:
        generation_data = GenerationData(
            generation_id=7,
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T00:30:00",
        )
        generation_data.add_episode(
            self.make_episode(
                0,
                row_count=2,
                success=True,
                duration_seconds=3.0,
                episode_execution_seconds=2.5,
                total_input_tokens=100,
                total_output_tokens=40,
                peak_context_tokens=70,
                avg_context_tokens=60.0,
                median_context_tokens=58.0,
            )
        )
        generation_data.add_episode(
            self.make_episode(
                1,
                row_count=4,
                success=True,
                duration_seconds=5.0,
                episode_execution_seconds=4.0,
                total_input_tokens=180,
                total_output_tokens=60,
                peak_context_tokens=90,
                avg_context_tokens=80.0,
                median_context_tokens=78.0,
            )
        )

        metadata = generation_data.to_metadata_dict(run_id="run-123")

        self.assertEqual(metadata["total_input_tokens"], 280)
        self.assertEqual(metadata["total_output_tokens"], 100)
        self.assertEqual(metadata["total_tokens"], 380)
        self.assertEqual(metadata["episodes_per_minute"], 2 / 30)
        self.assertEqual(metadata["training_rows_per_minute"], 6 / 30)
        self.assertEqual(metadata["total_agent_seconds"], 8.0)
        self.assertEqual(metadata["total_episode_execution_seconds"], 6.5)
        self.assertEqual(metadata["peak_context_tokens"], 90)
        self.assertEqual(metadata["avg_context_tokens"], 70.0)
        self.assertEqual(metadata["median_context_tokens"], 68.0)
        self.assertEqual(metadata["tokens_per_successful_episode"], 190.0)

    def test_generation_metadata_omits_wall_clock_when_timestamps_missing(self) -> None:
        generation_data = GenerationData(generation_id=7)
        generation_data.add_episode(self.make_episode(0, success=True))

        metadata = generation_data.to_metadata_dict(run_id="run-123")

        self.assertNotIn("generation_wall_clock_seconds", metadata)


if __name__ == "__main__":
    unittest.main()
