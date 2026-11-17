"""Tests for guard.py. Run from the repo root:

    uv run python -m unittest discover -s .claude/hooks -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402

WINDOWS = sys.platform == "win32"


class GuardTestCase(unittest.TestCase):
    whitelist: list[str] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        (cls.root / "build").mkdir()
        (cls.root / "src").mkdir()
        cls.policy = guard.load_policy(sys.platform, cls.root, whitelist_override=cls.whitelist)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def decide(self, command: str, tool: str = "Bash", cwd: Path | None = None) -> int:
        level, _ = guard.evaluate(tool, command, str(cwd or self.root), self.policy)
        return level

    def assertDecision(self, expected: int, command: str, tool: str = "Bash", cwd: Path | None = None):
        level, reasons = guard.evaluate(tool, command, str(cwd or self.root), self.policy)
        self.assertEqual(expected, level, f"{tool}: {command!r} -> {reasons}")


class GitPolicy(GuardTestCase):
    def test_read_only_git_passes(self):
        for cmd in ["git status", "git log --oneline -5", "git diff HEAD~1", "git show HEAD",
                    "git blame README.md", "git rev-parse HEAD", "git ls-files", "git describe",
                    "git shortlog -s", "git reflog", "git branch", "git branch -a", "git branch --list 'f*'",
                    "git tag", "git tag -l 'v*'", "git remote -v", "git remote show origin",
                    "git stash list", "git config --get user.name", "git -C src status",
                    "git status && git log", "git --no-pager log"]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.NONE, cmd)

    def test_acting_git_asks(self):
        for cmd in ["git commit -m x", "git push", "git merge main", "git rebase main", "git reset --hard",
                    "git checkout main", "git switch dev", "git restore x", "git stash", "git tag v1",
                    "git branch new", "git branch -D old", "git add .", "git pull", "git fetch",
                    "git init", "git clone url", "git config user.name me", "git remote add o url",
                    "git cherry-pick abc", "git revert abc", "git am x.patch", "git apply x.patch",
                    "git worktree add ../w", "git submodule update", "git gc", "git frobnicate",
                    "git", "git --version", "git diff --output=out.txt"]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.ASK, cmd)

    def test_git_hidden_in_compound_or_wrapped_commands_asks(self):
        for cmd in ["cd src && git -C . push", "git -C src commit -m x", "echo hi; git push",
                    "ls | git commit -F -", "bash -c 'git push'", "sh -c \"git commit -m x\"",
                    "eval git push", "sudo git push", "env FOO=1 git push", "GIT_DIR=x git status",
                    "git -c core.pager=less log", "echo \"$(git push)\"", "timeout 5 git push",
                    "git $(echo push)", "python -c \"import os; os.system('git push')\"",
                    "cmd //c git push" if WINDOWS else "nohup git push"]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.ASK, cmd)

    def test_powershell_git(self):
        self.assertDecision(guard.NONE, "git status; git log -3", tool="PowerShell")
        self.assertDecision(guard.ASK, "git commit -m 'x'", tool="PowerShell")
        self.assertDecision(guard.ASK, "Set-Location src; git push", tool="PowerShell")
        self.assertDecision(guard.ASK, "powershell -NoProfile -Command \"git push\"", tool="PowerShell")
        self.assertDecision(guard.ASK, "& git push", tool="PowerShell")

    def test_comment_is_not_a_command(self):
        self.assertDecision(guard.NONE, "git status # then git push")


class DeleteWithEmptyWhitelist(GuardTestCase):
    whitelist = []

    def test_recursive_deletes_are_denied(self):
        for cmd in ["rm -rf build", "rm -r build", "rm -fr build", "rm -R build", "rm --recursive build",
                    "rm --rec build", "rm -r -f build", "/bin/rm -rf build", "\\rm -rf build",
                    "cd src && rm -rf ../build", "sudo rm -rf build", "xargs rm -rf < list",
                    "find build -delete", "find . -name '*.o' -exec rm {} \\;",
                    "find . -exec sh -c 'rm -rf x' \\;", "git clean -fdx", "git clean -fd src",
                    "npx rimraf build", "rimraf build", "bash -c 'rm -rf build'",
                    "python -c \"import shutil; shutil.rmtree('build')\"", "rm -rf $BUILD",
                    "rm -rf build/*", "rm -rf ~/x", "echo \"$(rm -rf build)\"", "rm -rf {a,b}",
                    "for d in a b; do rm -rf $d; done", "$(which rm) -rf build", "eval rm -rf build",
                    "rm -rf \"unterminated", "uv run python -c \"import shutil; shutil.rmtree('b')\""]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.DENY, cmd)

    def test_non_recursive_and_safe_commands_pass(self):
        for cmd in ["rm file.txt", "rm -f build/a.o", "rmdir empty", "find . -name '*.c'", "ls -la",
                    "git clean -n -d", "echo rm -rf build", "mkdir -p build", "uv run pytest",
                    "npm install rimraf", "git commit -m 'remove rmtree usage'"]:
            with self.subTest(cmd=cmd):
                expected = guard.ASK if cmd.startswith("git") and "clean" not in cmd else guard.NONE
                if cmd.startswith("git clean"):
                    expected = guard.ASK  # git clean is not read-only, but a dry run is not a delete
                self.assertDecision(expected, cmd)

    def test_powershell_deletes_are_denied(self):
        for cmd in ["Remove-Item -Recurse build", "Remove-Item build -Recurse -Force", "rm -r build",
                    "ri -rec build", "Remove-Item -Path build -Recurse", "Remove-Item -Path:build -Recurse",
                    "Get-ChildItem build -Recurse | Remove-Item", "gci build | rm",
                    "Remove-Item (Join-Path $PWD build) -Recurse", "Remove-Item $env:TEMP\\x -Recurse",
                    "[IO.Directory]::Delete('build', $true)", "Invoke-Expression $cmd",
                    "powershell -EncodedCommand AAAA", "cmd /c rd /s /q build" if WINDOWS else "rm -rf build",
                    "Set-Location src; Remove-Item ..\\build -Recurse"]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.DENY, cmd, tool="PowerShell")

    def test_powershell_safe_commands_pass(self):
        for cmd in ["Remove-Item build\\a.o", "Get-ChildItem -Recurse", "Write-Output 'rm -rf build'",
                    "$x = 1; $x.Count", "New-Item -ItemType Directory build"]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.NONE, cmd, tool="PowerShell")


class DeleteWithWhitelist(GuardTestCase):
    whitelist = ["build"]

    def test_deletes_inside_whitelist_pass(self):
        for cmd in ["rm -rf build", "rm -rf ./build", "rm -rf build/obj", "cd build && rm -rf obj",
                    "find build -delete", "npx rimraf build/tmp", "git clean -fdx build"]:
            with self.subTest(cmd=cmd):
                expected = guard.ASK if cmd.startswith("git") else guard.NONE
                self.assertDecision(expected, cmd)
        self.assertDecision(guard.NONE, "Remove-Item -Recurse -Force build", tool="PowerShell")

    def test_deletes_escaping_whitelist_are_denied(self):
        for cmd in ["rm -rf build/..", "rm -rf build/../src", "rm -rf build src", "rm -rf .",
                    "cd build; rm -rf obj", "(cd build && rm -rf obj); rm -rf obj", "git clean -fdx",
                    "find . -delete", "rm -rf ../build"]:
            with self.subTest(cmd=cmd):
                self.assertDecision(guard.DENY, cmd)
        # Relative to a different working directory.
        self.assertDecision(guard.DENY, "rm -rf build", cwd=self.root / "src")

    def test_absolute_paths(self):
        inside = (self.root / "build" / "x").as_posix()
        outside = (self.root / "src").as_posix()
        self.assertDecision(guard.NONE, f"rm -rf '{inside}'")
        self.assertDecision(guard.DENY, f"rm -rf '{outside}'")
        if WINDOWS:
            msys = "/" + inside[0].lower() + inside[2:]
            self.assertDecision(guard.NONE, f"rm -rf '{msys}'")


class PolicyValidation(unittest.TestCase):
    def test_bad_whitelist_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for entry in ["", "/abs", "C:/abs", "..", "../outside", ".", "build/*", ".git/x", "$HOME"]:
                with self.subTest(entry=entry):
                    with self.assertRaises(guard.PolicyError):
                        guard.load_policy(sys.platform, Path(tmp), whitelist_override=[entry])

    def test_unsupported_os_fails_closed(self):
        with self.assertRaises(guard.PolicyError):
            guard.load_policy("darwin")

    def test_shipped_rule_files_load(self):
        for os_key in guard.RULE_FILES:
            with self.subTest(os=os_key):
                policy = guard.load_policy(os_key)
                self.assertEqual((), policy.whitelist)


class HookProtocol(unittest.TestCase):
    """End-to-end: JSON on stdin, decision JSON on stdout."""

    def run_hook(self, payload: dict) -> dict | None:
        proc = subprocess.run([sys.executable, str(Path(guard.__file__))], input=json.dumps(payload),
                              capture_output=True, text=True, check=True)
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def test_protocol(self):
        cwd = str(guard.PROJECT_DIR)
        self.assertIsNone(self.run_hook({"tool_name": "Bash", "tool_input": {"command": "git status"}, "cwd": cwd}))
        out = self.run_hook({"tool_name": "Bash", "tool_input": {"command": "git push"}, "cwd": cwd})
        self.assertEqual("ask", out["hookSpecificOutput"]["permissionDecision"])
        out = self.run_hook({"tool_name": "PowerShell", "tool_input": {"command": "rm -r x"}, "cwd": cwd})
        self.assertEqual("deny", out["hookSpecificOutput"]["permissionDecision"])
        self.assertIsNone(self.run_hook({"tool_name": "Read", "tool_input": {"file_path": "x"}}))

    def test_garbage_input_fails_closed(self):
        proc = subprocess.run([sys.executable, str(Path(guard.__file__))], input="not json",
                              capture_output=True, text=True, check=True)
        self.assertEqual("deny", json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"])


if __name__ == "__main__":
    unittest.main()
