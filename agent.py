import subprocess
import sys
from pathlib import Path

import anthropic

import prompts as _default_prompts
from tools import TOOL_SCHEMAS, dispatch_tool

DEFAULT_MODEL = "claude-sonnet-4-6"


class AgentLoop:
    def __init__(
        self,
        client: anthropic.Anthropic,
        task_dir: Path,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 10,
        prompts_module=None,
    ) -> None:
        self.client = client
        self.task_dir = task_dir
        self.model = model
        self.max_iterations = max_iterations
        self._prompts = prompts_module if prompts_module is not None else _default_prompts

    # ------------------------------------------------------------------
    # Phase 1: generate locked test file
    # ------------------------------------------------------------------

    def generate_tests(self, user_prompt: str) -> str:
        print("Generating test file...", flush=True)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self._prompts.TEST_GENERATION_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": self._prompts.TEST_GENERATION_USER.format(user_prompt=user_prompt),
                }
            ],
        )
        test_content = response.content[0].text.strip()
        test_path = self.task_dir / "solution_test.py"
        test_path.write_text(test_content)
        print(f"Test file written to: {test_path}", flush=True)
        return test_content

    # ------------------------------------------------------------------
    # Phase 2: iterative implementation loop
    # ------------------------------------------------------------------

    def run_implementation_loop(self, user_prompt: str, test_content: str) -> tuple[bool, str]:
        prompt_md_path = self.task_dir / "solution.prompt.md"
        prompt_md_section = ""
        if prompt_md_path.exists():
            prompt_md_section = self._prompts.PROMPT_MD_SECTION.format(
                prompt_md=prompt_md_path.read_text()
            )

        user_content = self._prompts.IMPLEMENTATION_USER.format(
            user_prompt=user_prompt,
            test_content=test_content,
            prompt_md_section=prompt_md_section,
        )

        messages: list[dict] = [{"role": "user", "content": user_content}]
        iteration = 0
        last_output = ""

        while iteration < self.max_iterations:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=self._prompts.IMPLEMENTATION_SYSTEM,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # Collect any tool calls in this response turn
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            # Track pytest output for termination detection
            for block in response.content:
                if block.type == "tool_use" and block.name == "run_subprocess":
                    # Result isn't available yet; we'll capture it after dispatch
                    pass

            if response.stop_reason == "end_turn":
                # Ground-truth check: run pytest ourselves regardless of what the LLM ran
                solution_path = self.task_dir / "solution.py"
                if solution_path.exists():
                    result = subprocess.run(
                        ["pytest", "solution_test.py", "-v"],
                        cwd=str(self.task_dir),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    final_output = result.stdout + result.stderr
                    if _tests_passed(final_output):
                        return True, final_output
                    last_output = final_output

                final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return False, f"Agent stopped without passing tests.\n{final_text}\n\n{last_output}"

            if not tool_calls:
                return False, "Agent stopped requesting tools before tests passed."

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Dispatch all tool calls, build tool_result list
            tool_results = []
            for tc in tool_calls:
                result_str = dispatch_tool(tc.name, tc.input, self.task_dir)

                if tc.name == "run_subprocess":
                    last_output = result_str
                    passed = _tests_passed(result_str)
                    status = "PASS" if passed else "FAIL"
                    print(f"  [iteration {iteration + 1}] pytest → {status}", flush=True)
                elif tc.name == "write_file":
                    print(f"  [iteration {iteration + 1}] writing {tc.input.get('path', '?')}", flush=True)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

            # Count a full write→run cycle as one iteration
            has_write = any(tc.name == "write_file" for tc in tool_calls)
            has_run = any(tc.name == "run_subprocess" for tc in tool_calls)
            if has_write and has_run:
                iteration += 1

        return False, f"Max iterations ({self.max_iterations}) reached.\n{last_output}"

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self, user_prompt: str, auto_approve: bool = False) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1
        test_content = self.generate_tests(user_prompt)

        # Show tests, optionally wait for approval
        print("\n" + "=" * 60)
        print("GENERATED TESTS (solution_test.py):")
        print("=" * 60)
        print(test_content)
        print("=" * 60)
        if not auto_approve:
            try:
                input("\nPress Enter to start implementation, or Ctrl+C to abort...\n")
            except KeyboardInterrupt:
                print("\nAborted.")
                sys.exit(0)

        # Phase 2
        print("Starting implementation loop...", flush=True)
        success, message = self.run_implementation_loop(user_prompt, test_content)

        print("\n" + "=" * 60)
        if success:
            print("SUCCESS — all tests passed!")
        else:
            print("FAILED — could not pass all tests.")
        print("=" * 60)
        print(message)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tests_passed(pytest_output: str) -> bool:
    """Check the pytest summary line, e.g. '12 passed in 0.03s'."""
    for line in reversed(pytest_output.splitlines()):
        line = line.strip()
        if "passed" in line and ("==" in line or line.startswith("PASSED")):
            return "failed" not in line and "error" not in line
    return False
