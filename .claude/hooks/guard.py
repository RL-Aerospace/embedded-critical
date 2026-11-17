"""PreToolUse guard for Claude Code's Bash and PowerShell tools.

Policy
------
1. Git: every git command that is not on the read-only allowlist below gets
   permissionDecision "ask" (the user must confirm it).
2. Recursive deletes: denied unless every target resolves inside a folder that
   is whitelisted in the rule file for the host OS (rules/windows.toml or
   rules/linux.toml). The rule files also define which commands count as
   recursive deletes on that OS.
3. Fail closed: whatever the guard cannot resolve is treated as dangerous.
   An unresolvable delete is denied, an unresolvable git command asks, and a
   broken rule file, an unsupported OS, or an internal error denies the call.

The guard only ever tightens permissions. It emits "ask" or "deny" and never
"allow", so normal permission handling still applies to everything it passes.

Known limits: it inspects the command text Claude sends. It cannot see inside
programs or scripts that the command runs (make, ./build.sh, python tool.py,
npm run ...), and it is not a sandbox.

Runs under uv (see the hook in .claude/settings.json); requires Python >= 3.12.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIN_PYTHON = (3, 12)
SCHEMA_VERSION = 1
MAX_DEPTH = 8

NONE, ASK, DENY = 0, 1, 2
DECISION_NAMES = {ASK: "ask", DENY: "deny"}

HOOK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HOOK_DIR.parents[1]
RULES_DIR = HOOK_DIR / "rules"
RULE_FILES = {"win32": "windows.toml", "linux": "linux.toml"}
TOOL_DIALECT = {"Bash": "bash", "PowerShell": "powershell"}

# --------------------------------------------------------------------------
# Git policy (OS independent). Anything not proven read-only asks.
# --------------------------------------------------------------------------
GIT_ALWAYS_READ_ONLY = frozenset(
    {"status", "log", "diff", "show", "blame", "rev-parse", "ls-files", "describe", "shortlog"}
)
# Global options that consume the following argument.
GIT_GLOBAL_OPTS_WITH_VALUE = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix"}
)
# Global options that can make git run arbitrary programs (pager, diff driver...).
GIT_RISKY_GLOBAL_OPTS = frozenset({"-c", "--config-env", "--exec-path"})

_BRANCH_TAG_VALUE_FLAGS = frozenset(
    {"--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format"}
)
GIT_BRANCH_LIST_FLAGS = frozenset(
    {"-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose", "--show-current", "--color",
     "--no-color", "--column", "--no-column", "-i", "--ignore-case", "--abbrev", "--no-abbrev",
     "-l", "--list", "--omit-empty"}
)
GIT_TAG_LIST_FLAGS = frozenset(
    {"-l", "--list", "--column", "--no-column", "--color", "--no-color", "-i", "--ignore-case",
     "--omit-empty"}
)
GIT_CONFIG_READ_FLAGS = frozenset(
    {"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--get-color", "--get-colorbool",
     "--list", "-l"}
)
GIT_CONFIG_WRITE_ARGS = frozenset(
    {"--add", "--unset", "--unset-all", "--replace-all", "--rename-section", "--remove-section",
     "-e", "--edit", "set", "unset", "rename-section", "remove-section", "edit"}
)

# --------------------------------------------------------------------------
# Shell vocabulary
# --------------------------------------------------------------------------
CD_COMMANDS = frozenset(
    {"cd", "chdir", "pushd", "popd", "set-location", "sl", "push-location", "pop-location"}
)
BASH_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "ash"})
PS_SHELLS = frozenset({"powershell", "pwsh"})
CMD_SHELLS = frozenset({"cmd"})
# Commands that run another command given later on their command line.
WRAPPERS = frozenset(
    {"sudo", "doas", "env", "nohup", "time", "command", "builtin", "exec", "nice", "ionice",
     "timeout", "stdbuf", "xargs", "chrt", "taskset", "watch", "unbuffer", "wsl", "start",
     "start-process", "saps", "npx", "bunx", "pnpx", "uvx"}
)
# Package managers that run another command only through specific subcommands.
PACKAGE_RUNNERS = {
    "npm": frozenset({"exec", "x"}),
    "pnpm": frozenset({"dlx", "exec"}),
    "yarn": frozenset({"dlx", "exec"}),
    "bun": frozenset({"x"}),
    "uv": frozenset({"run", "tool"}),
}
BASH_RESERVED = frozenset({"!", "then", "else", "elif", "do", "if", "while", "until", "time", "coproc"})
PS_RESERVED = frozenset({"if", "elseif", "else", "while", "until", "do", "foreach", "for", "switch",
                         "try", "catch", "finally", "return", "throw"})
ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=.*", re.DOTALL)
DYNAMIC_CHARS = frozenset("$%*?[]~`{}!")
UNKNOWN = object()  # sentinel for "cannot be determined"

DELETE_PARSERS = frozenset({"posix-rm", "powershell-remove", "cmd-switch", "find", "targets"})
ALL_DIALECTS = frozenset({"bash", "powershell", "cmd"})


class PolicyError(Exception):
    """The rule file is missing or invalid; the guard must fail closed."""


class LexError(ValueError):
    """The command text could not be tokenized (e.g. unterminated quote)."""


# --------------------------------------------------------------------------
# Policy (rule file) loading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeleteRule:
    name: str
    dialects: frozenset[str]
    commands: frozenset[str]
    parser: str
    opts: dict


@dataclass(frozen=True)
class CommandPattern:
    name: str
    dialects: frozenset[str]
    regex: re.Pattern


@dataclass
class Policy:
    os_key: str
    rule_file: Path
    project_dir: Path
    whitelist: tuple[Path, ...]
    delete_rules: tuple[DeleteRule, ...]
    git_clean_flags: frozenset[str]
    interpreters: frozenset[str]
    inline_patterns: tuple[re.Pattern, ...]
    command_patterns: tuple[CommandPattern, ...]

    @property
    def windows(self) -> bool:
        return self.os_key == "win32"

    def delete_names(self, dialect: str) -> frozenset[str]:
        return frozenset(c for r in self.delete_rules if dialect in r.dialects for c in r.commands)

    def resolve(self, raw: str, base: Path | None, dialect: str) -> Path | None:
        """Absolute, normalized path for raw (relative to base); None if base unknown."""
        text = raw
        if self.windows and dialect == "bash":
            m = re.match(r"^/([A-Za-z])(?:/(.*))?$", text, re.DOTALL)  # Git Bash /c/... form
            if m:
                text = f"{m.group(1)}:/{m.group(2) or ''}"
        path = Path(text)
        if not path.is_absolute():
            if base is None:
                return None
            path = base / path
        return Path(os.path.realpath(path))

    def in_whitelist(self, path: Path) -> bool:
        target = os.path.normcase(str(path))
        for folder in self.whitelist:
            root = os.path.normcase(str(folder)).rstrip("\\/")
            if target == root or target.startswith(root + os.sep):
                return True
        return False


def _str_list(table: dict, key: str, where: str, required: bool = True) -> list[str]:
    value = table.get(key, None if required else [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError(f"{where}: '{key}' must be a list of strings")
    return value


def load_policy(os_key: str, project_dir: Path = PROJECT_DIR,
                whitelist_override: list[str] | None = None) -> Policy:
    import tomllib

    file_name = RULE_FILES.get(os_key)
    if file_name is None:
        raise PolicyError(f"unsupported OS '{os_key}' (supported: windows, linux)")
    rule_file = RULES_DIR / file_name
    try:
        data = tomllib.loads(rule_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PolicyError(f"rule file {rule_file} is missing") from None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PolicyError(f"rule file {rule_file} cannot be read: {exc}") from None

    where = rule_file.name
    if data.get("schema") != SCHEMA_VERSION:
        raise PolicyError(f"{where}: 'schema' must be {SCHEMA_VERSION}")
    expected_os = file_name.removesuffix(".toml")
    if data.get("os") != expected_os:
        raise PolicyError(f"{where}: 'os' must be \"{expected_os}\"")

    project_real = Path(os.path.realpath(project_dir))
    entries = whitelist_override if whitelist_override is not None else _str_list(
        data.get("whitelist", {}), "paths", f"{where} [whitelist]")
    whitelist = []
    for entry in entries:
        if not entry.strip() or Path(entry).is_absolute() or Path(entry).drive or entry.startswith(("/", "\\")):
            raise PolicyError(f"{where}: whitelist entry {entry!r} must be a non-empty repo-relative path")
        if any(ch in entry for ch in "*?[]$%~"):
            raise PolicyError(f"{where}: whitelist entry {entry!r} must not contain wildcards or variables")
        resolved = Path(os.path.realpath(project_real / entry))
        root = os.path.normcase(str(project_real))
        res = os.path.normcase(str(resolved))
        if res == root or not res.startswith(root.rstrip("\\/") + os.sep):
            raise PolicyError(f"{where}: whitelist entry {entry!r} must be a folder inside the repo, not the repo itself")
        if os.path.normcase(".git") in Path(res).relative_to(root).parts:
            raise PolicyError(f"{where}: whitelist entry {entry!r} must not be inside .git")
        whitelist.append(resolved)

    rules = []
    for idx, raw in enumerate(data.get("delete", [])):
        rwhere = f"{where} [[delete]] #{idx + 1}"
        name = raw.get("name")
        parser = raw.get("parser")
        if not isinstance(name, str) or not name:
            raise PolicyError(f"{rwhere}: 'name' is required")
        if parser not in DELETE_PARSERS:
            raise PolicyError(f"{rwhere}: 'parser' must be one of {sorted(DELETE_PARSERS)}")
        dialects = frozenset(_str_list(raw, "dialects", rwhere))
        if not dialects or not dialects <= ALL_DIALECTS:
            raise PolicyError(f"{rwhere}: 'dialects' must be a non-empty subset of {sorted(ALL_DIALECTS)}")
        commands = frozenset(c.lower() for c in _str_list(raw, "commands", rwhere))
        opts = {k: v for k, v in raw.items() if k not in ("name", "parser", "dialects", "commands")}
        required = {
            "posix-rm": ("recursive_short", "recursive_long"),
            "powershell-remove": ("recursive_params", "path_params", "value_params"),
            "cmd-switch": ("recursive_switches",),
            "find": ("delete_actions", "exec_actions", "exec_delete_commands", "exec_opaque_commands"),
            "targets": (),
        }[parser]
        for key in required:
            opts[key] = [v.lower() if parser != "posix-rm" else v for v in _str_list(raw, key, rwhere)]
        rules.append(DeleteRule(name, dialects, commands, parser, opts))
    if not rules:
        raise PolicyError(f"{where}: at least one [[delete]] rule is required")

    git_clean = frozenset(_str_list(data.get("git_clean", {}), "recursive_flags", f"{where} [git_clean]"))
    inline = data.get("inline_code", {})
    interpreters = frozenset(i.lower() for i in _str_list(inline, "interpreters", f"{where} [inline_code]"))
    try:
        inline_patterns = tuple(re.compile(p, re.IGNORECASE)
                                for p in _str_list(inline, "patterns", f"{where} [inline_code]"))
        command_patterns = tuple(
            CommandPattern(p["name"], frozenset(p["dialects"]), re.compile(p["regex"], re.IGNORECASE))
            for p in data.get("command_pattern", [])
        )
    except (re.error, KeyError, TypeError) as exc:
        raise PolicyError(f"{where}: invalid pattern: {exc}") from None

    return Policy(os_key, rule_file, project_real, tuple(whitelist), tuple(rules), git_clean,
                  interpreters, inline_patterns, command_patterns)


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Token:
    text: str
    op: bool


OPERATOR_CHARS = {
    "bash": frozenset(";&|(){}<>`\n"),
    "powershell": frozenset(";&|(){}<>\n"),
    "cmd": frozenset("&|()<>\n"),
}
ESCAPE_CHAR = {"bash": "\\", "powershell": "`", "cmd": "^"}


def lex(text: str, dialect: str) -> list[Token]:
    """Split a command line into words and operator runs, honouring quotes."""
    ops = OPERATOR_CHARS[dialect]
    esc = ESCAPE_CHAR[dialect]
    tokens: list[Token] = []
    buf: list[str] = []
    in_word = False
    i, n = 0, len(text)

    def flush() -> None:
        nonlocal in_word
        if in_word:
            tokens.append(Token("".join(buf), False))
        buf.clear()
        in_word = False

    while i < n:
        c = text[i]
        if c == esc:
            if i + 1 < n:
                if text[i + 1] != "\n":  # escaped newline = line continuation
                    buf.append(text[i + 1])
                    in_word = True
                i += 2
            else:
                buf.append(c)
                in_word = True
                i += 1
            continue
        if c == "'" and dialect != "cmd":
            j, part = i + 1, []
            while True:
                if j >= n:
                    raise LexError("unterminated single quote")
                if text[j] == "'":
                    if dialect == "powershell" and j + 1 < n and text[j + 1] == "'":
                        part.append("'")
                        j += 2
                        continue
                    break
                part.append(text[j])
                j += 1
            buf.append("".join(part))
            in_word = True
            i = j + 1
            continue
        if c == '"':
            j, part = i + 1, []
            while True:
                if j >= n:
                    raise LexError("unterminated double quote")
                ch = text[j]
                if ch == '"':
                    if dialect == "powershell" and j + 1 < n and text[j + 1] == '"':
                        part.append('"')
                        j += 2
                        continue
                    break
                if dialect == "bash" and ch == "\\" and j + 1 < n and text[j + 1] in '$`"\\\n':
                    if text[j + 1] != "\n":
                        part.append(text[j + 1])
                    j += 2
                    continue
                if dialect == "powershell" and ch == "`" and j + 1 < n:
                    part.append(text[j + 1])
                    j += 2
                    continue
                part.append(ch)
                j += 1
            buf.append("".join(part))
            in_word = True
            i = j + 1
            continue
        if c in ops:
            if c in "<>" and in_word and re.fullmatch(r"\d+|\*", "".join(buf)):
                buf.clear()  # file-descriptor prefix such as 2> or *>
                in_word = False
            flush()
            if c == "\n":
                tokens.append(Token("\n", True))
                i += 1
                continue
            j = i
            while j < n and text[j] in ops and text[j] != "\n":
                j += 1
            tokens.append(Token(text[i:j], True))
            i = j
            continue
        if c.isspace():
            flush()
            i += 1
            continue
        if c == "#" and not in_word and dialect != "cmd":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        buf.append(c)
        in_word = True
        i += 1
    flush()
    return tokens


def is_redirect(op: str) -> bool:
    return set(op) <= set("<>&") and ("<" in op or ">" in op)


@dataclass
class Segment:
    words: list[str]
    sep_before: str
    sep_after: str


def split_segments(tokens: list[Token]) -> list[Segment]:
    segments: list[Segment] = []
    current: list[str] = []
    prev = ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.op and is_redirect(tok.text):
            i += 2 if i + 1 < len(tokens) and not tokens[i + 1].op else 1
            continue
        if tok.op:
            segments.append(Segment(current, prev, tok.text))
            current, prev = [], tok.text
        else:
            current.append(tok.text)
        i += 1
    segments.append(Segment(current, prev, ""))
    return segments


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def cmd_name(word: str) -> str:
    base = re.split(r"[\\/]", word)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def is_dynamic(word: str) -> bool:
    return word == "" or any(ch in DYNAMIC_CHARS for ch in word)


def ps_param_matches(token: str, params: list[str]) -> bool:
    """PowerShell accepts any unambiguous prefix of a parameter name (e.g. -Rec)."""
    return len(token) >= 2 and any(p.startswith(token) for p in params)


def substitutions(word: str, dialect: str) -> list[str]:
    """Command substitutions embedded in a (double-quoted) word."""
    found: list[str] = []
    if dialect in ("bash", "powershell"):
        start = 0
        while (k := word.find("$(", start)) != -1:
            depth, j = 0, k + 1
            while j < len(word):
                if word[j] == "(":
                    depth += 1
                elif word[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            found.append(word[k + 2 : j])
            start = j + 1
    if dialect == "bash" and "`" in word:
        found.extend(word.split("`")[1::2])
    return found


def parse_posix_rm(args: list[str], opts: dict) -> tuple[bool, list[str]]:
    recursive, targets, end_of_opts = False, [], False
    for a in args:
        if not end_of_opts and a == "--":
            end_of_opts = True
        elif not end_of_opts and a.startswith("--"):
            name = a.split("=", 1)[0]
            # GNU getopt accepts unambiguous prefixes, e.g. --rec for --recursive.
            if len(name) >= 3 and any(lo.startswith(name) for lo in opts["recursive_long"]):
                recursive = True
        elif not end_of_opts and a.startswith("-") and len(a) > 1:
            if any(ch in opts["recursive_short"] for ch in a[1:]):
                recursive = True
        else:
            targets.append(a)
    return recursive, targets


def parse_ps_remove(args: list[str], opts: dict) -> tuple[bool, list[str]]:
    recursive, targets, i = False, [], 0
    while i < len(args):
        a = args[i]
        if a.startswith("-") and len(a) > 1:
            name, colon, inline = a.partition(":")
            low = name.lower()
            if ps_param_matches(low, opts["recursive_params"]):
                recursive = True  # -Recurse:$false is still treated as recursive (fail closed)
            elif ps_param_matches(low, opts["path_params"]):
                if colon:
                    targets.extend(inline.split(","))
                elif i + 1 < len(args):
                    targets.extend(args[i + 1].split(","))
                    i += 1
                else:
                    targets.append("")
            elif ps_param_matches(low, opts["value_params"]) and not colon:
                i += 1
        else:
            targets.extend(a.split(","))
        i += 1
    return recursive, targets


def parse_cmd_switch(args: list[str], opts: dict) -> tuple[bool, list[str]]:
    recursive, targets = False, []
    for a in args:
        if a.startswith("/"):
            if any("/" + part.lower() in opts["recursive_switches"] for part in a.split("/") if part):
                recursive = True
        else:
            targets.append(a)
    return recursive, targets


def parse_find(args: list[str], opts: dict) -> tuple[bool, list[str], str | None]:
    i = 0
    while i < len(args) and args[i] in ("-H", "-L", "-P"):
        i += 1
    while i < len(args) and (args[i] == "-D" or args[i].startswith("-O")):
        i += 2 if args[i] == "-D" else 1
    paths = []
    while i < len(args) and not args[i].startswith("-") and args[i] not in ("(", ")", "!", ","):
        paths.append(args[i])
        i += 1
    expr = args[i:]
    deleting, opaque = False, None
    for j, tok in enumerate(expr):
        low = tok.lower()
        if low in opts["delete_actions"]:
            deleting = True
        elif low in opts["exec_actions"] and j + 1 < len(expr):
            run = cmd_name(expr[j + 1])
            if run in opts["exec_delete_commands"]:
                deleting = True
            elif run in opts["exec_opaque_commands"]:
                opaque = run
    return deleting, paths or ["."], opaque


def _git_listing_only(args: list[str], flags: frozenset[str], value_flags: frozenset[str],
                      allow_count_flag: bool = False) -> bool:
    positional_ok = "-l" in args or "--list" in args
    i = 0
    while i < len(args):
        a = args[i]
        name = a.split("=", 1)[0]
        if name in value_flags:
            if "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
        elif name in flags or (allow_count_flag and re.fullmatch(r"-n\d*", a)):
            pass
        elif a.startswith("-") or not positional_ok:
            return False
        i += 1
    return True


def git_is_read_only(sub: str, rest: list[str]) -> bool:
    if any(a.startswith("--output") for a in rest):
        return False  # e.g. git diff --output=<file> writes a file
    if sub in GIT_ALWAYS_READ_ONLY:
        return True
    if sub == "reflog":
        return not rest or rest[0] == "show" or rest[0].startswith("-")
    if sub == "branch":
        return _git_listing_only(rest, GIT_BRANCH_LIST_FLAGS, _BRANCH_TAG_VALUE_FLAGS)
    if sub == "tag":
        return _git_listing_only(rest, GIT_TAG_LIST_FLAGS, _BRANCH_TAG_VALUE_FLAGS, allow_count_flag=True)
    if sub == "remote":
        tail = [a for a in rest if a not in ("-v", "--verbose")]
        return not tail or tail[0] in ("show", "get-url")
    if sub == "stash":
        return bool(rest) and rest[0] in ("list", "show")
    if sub == "config":
        if any(a.split("=", 1)[0] in GIT_CONFIG_WRITE_ARGS for a in rest):
            return False
        return any(a.split("=", 1)[0] in GIT_CONFIG_READ_FLAGS for a in rest) or (
            bool(rest) and rest[0] in ("get", "list"))
    return False


# --------------------------------------------------------------------------
# Analyzer
# --------------------------------------------------------------------------
@dataclass
class State:
    cwds: frozenset  # possible working directories (Path, or None when unknown)
    stack: list = field(default_factory=list)


class Analyzer:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.level = NONE
        self.reasons: list[str] = []

    def flag(self, level: int, reason: str) -> None:
        self.level = max(self.level, level)
        if reason not in self.reasons:
            self.reasons.append(reason)

    # -- entry -------------------------------------------------------------
    def run(self, text: str, dialect: str, cwds: frozenset, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            self.flag(DENY, "command nesting is too deep to inspect")
            return
        try:
            tokens = lex(text, dialect)
        except LexError as exc:
            self._fallback(text, dialect, str(exc))
            return
        state = State(cwds)
        for seg in split_segments(tokens):
            self._segment(seg, dialect, state, depth)
            for ch in seg.sep_after:
                if ch in "({":
                    state.stack.append(state.cwds)
                elif ch in ")}" and state.stack:
                    state.cwds = state.stack.pop() | state.cwds  # conservative: either may apply

    def _fallback(self, text: str, dialect: str, why: str) -> None:
        words = set(re.findall(r"[a-z0-9_.\-]+", text.lower()))
        if words & self.policy.delete_names(dialect) or any(p.search(text) for p in self.policy.inline_patterns):
            self.flag(DENY, f"command could not be parsed ({why}) and mentions a delete command")
        if "git" in words:
            self.flag(ASK, f"command could not be parsed ({why}) and mentions git")

    # -- one simple command --------------------------------------------------
    def _segment(self, seg: Segment, dialect: str, state: State, depth: int) -> None:
        words = list(seg.words)
        for w in words:
            for inner in substitutions(w, dialect):
                self.run(inner, dialect, state.cwds, depth + 1)
        piped_in = "|" in seg.sep_before.replace("||", "")
        ps_call = dialect == "powershell" and seg.sep_before.endswith("&") and not seg.sep_before.endswith("&&")
        self._command(words, dialect, state, depth, sep_after=seg.sep_after, piped_in=piped_in,
                      ps_call=ps_call, stdin_targets=False, top=True)

    def _command(self, words: list[str], dialect: str, state: State, depth: int, *, sep_after: str,
                 piped_in: bool, ps_call: bool, stdin_targets: bool, top: bool) -> None:
        env_assign = False
        while words and dialect == "bash" and ASSIGNMENT_RE.fullmatch(words[0]):
            env_assign = True
            words = words[1:]
        if dialect == "powershell" and len(words) >= 2 and words[0].startswith("$") and words[1] in ("=", "+=", "-="):
            words = words[2:]
        reserved = BASH_RESERVED if dialect == "bash" else PS_RESERVED if dialect == "powershell" else frozenset()
        while words and words[0].lower() in reserved:
            words = words[1:]
        if dialect == "powershell" and words and words[0] == ".":
            ps_call, words = True, words[1:]
        if not words:
            return
        cmd, args = words[0], words[1:]
        if dialect == "cmd":
            cmd = cmd.lstrip("@")
        if self._dynamic_command(cmd, dialect, ps_call):
            self.flag(DENY, f"command name {cmd!r} is computed at run time and cannot be inspected")
            return
        name = cmd_name(cmd)
        continues_in_group = sep_after[:1] in ("(", "{")

        if name in CD_COMMANDS:
            if top:
                self._cd(name, args, dialect, state, sep_after, continues_in_group)
            return
        if name == "git":
            self._git(args, dialect, state, env_assign, continues_in_group)
        elif name in BASH_SHELLS:
            for k, a in enumerate(args):
                if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", a):
                    if k + 1 < len(args):
                        self.run(args[k + 1], "bash", state.cwds, depth + 1)
                    break
        elif name in PS_SHELLS:
            self._powershell_cli(args, state, depth)
        elif name in CMD_SHELLS:
            for k, a in enumerate(args):
                if a.lower() in ("/c", "/k", "//c", "//k"):
                    self.run(" ".join(args[k + 1:]), "cmd", state.cwds, depth + 1)
                    break
        elif name == "eval" and dialect == "bash":
            self.run(" ".join(args), "bash", state.cwds, depth + 1)
        elif name in ("iex", "invoke-expression") and dialect == "powershell":
            code = " ".join(args)
            if "$" in code or continues_in_group or piped_in:
                self.flag(DENY, "Invoke-Expression with computed code cannot be inspected")
            else:
                self.run(code, "powershell", state.cwds, depth + 1)
        elif name in WRAPPERS or (name in PACKAGE_RUNNERS and self._runner_sub(name, args)):
            self._wrapper(name, args, dialect, state, depth, sep_after, piped_in, stdin_targets)

        if name in self.policy.interpreters:
            for a in args:
                for pat in self.policy.inline_patterns:
                    if pat.search(a):
                        self.flag(DENY, f"code passed to {name} matches {pat.pattern!r} (recursive delete); "
                                        "its targets cannot be verified")
                if re.search(r"\bgit\b", a):
                    self.flag(ASK, f"code passed to {name} may run git")

        for rule in self.policy.delete_rules:
            if dialect in rule.dialects and name in rule.commands:
                self._delete(rule, name, args, dialect, state, piped_in, stdin_targets, continues_in_group)

        for pat in self.policy.command_patterns:
            if dialect in pat.dialects and pat.regex.search(cmd):
                self.flag(DENY, f"{pat.name}: {cmd!r} deletes directly through .NET/API calls; "
                                "targets cannot be verified")

    # -- pieces --------------------------------------------------------------
    @staticmethod
    def _dynamic_command(cmd: str, dialect: str, ps_call: bool) -> bool:
        if dialect == "bash":
            return "$" in cmd or "`" in cmd
        if dialect == "cmd":
            return "%" in cmd or "!" in cmd
        return ps_call and ("$" in cmd or cmd == "")

    @staticmethod
    def _runner_sub(name: str, args: list[str]) -> bool:
        first = next((a for a in args if not a.startswith("-")), None)
        return first is not None and first.lower() in PACKAGE_RUNNERS[name]

    def _wrapper(self, name: str, args: list[str], dialect: str, state: State, depth: int,
                 sep_after: str, piped_in: bool, stdin_targets: bool) -> None:
        interesting = ({"git", "eval", "iex", "invoke-expression"} | BASH_SHELLS | PS_SHELLS | CMD_SHELLS
                       | WRAPPERS | set(PACKAGE_RUNNERS) | self.policy.interpreters
                       | self.policy.delete_names(dialect))
        for k, a in enumerate(args):
            if cmd_name(a) in interesting and not (name in PACKAGE_RUNNERS and a.lower() in PACKAGE_RUNNERS[name]):
                self._command(args[k:], dialect, state, depth + 1, sep_after=sep_after, piped_in=piped_in,
                              ps_call=False, stdin_targets=stdin_targets or name == "xargs", top=False)
                return

    def _powershell_cli(self, args: list[str], state: State, depth: int) -> None:
        i = 0
        while i < len(args):
            a = args[i]
            low = a.lower()
            if low.startswith("-") and len(low) > 1:
                p = low.lstrip("-")
                if p in ("e", "ec", "en", "enc") or (len(p) >= 2 and "encodedcommand".startswith(p)):
                    self.flag(DENY, "powershell -EncodedCommand cannot be inspected")
                    return
                if "command".startswith(p):
                    code = " ".join(args[i + 1:])
                    if code.strip() == "-":
                        self.flag(DENY, "powershell reading commands from stdin cannot be inspected")
                    else:
                        self.run(code, "powershell", state.cwds, depth + 1)
                    return
                if "file".startswith(p):
                    return  # runs a script file: outside what the guard can see
                if p in ("ep", "ex") or (len(p) >= 2 and "executionpolicy".startswith(p)) or \
                        p in ("windowstyle", "w", "configurationname", "outputformat", "of", "inputformat", "if"):
                    i += 2
                    continue
                i += 1
                continue
            if a == "-":
                self.flag(DENY, "powershell reading commands from stdin cannot be inspected")
                return
            self.run(" ".join(args[i:]), "powershell", state.cwds, depth + 1)  # implicit -Command
            return

    def _cd(self, name: str, args: list[str], dialect: str, state: State, sep_after: str,
            continues_in_group: bool) -> None:
        target = UNKNOWN
        if name not in ("popd", "pop-location") and not continues_in_group:
            i = 0
            while i < len(args):
                a = args[i]
                if a == "-":
                    break
                if dialect == "powershell" and a.startswith("-"):
                    pname, colon, inline = a.partition(":")
                    if ps_param_matches(pname.lower(), ["-path", "-literalpath", "-lp", "-pspath"]):
                        target = inline if colon else (args[i + 1] if i + 1 < len(args) else UNKNOWN)
                        break
                    i += 1
                    continue
                if (dialect == "bash" and a.startswith("-")) or (dialect == "cmd" and a.startswith("/")):
                    i += 1
                    continue
                target = a
                break
        if target is UNKNOWN or is_dynamic(target):
            new = frozenset({None})
        else:
            new = frozenset(self.policy.resolve(target, c, dialect) for c in state.cwds)
        # Only "cd X && ..." guarantees the later commands run in X.
        state.cwds = new if sep_after == "&&" else state.cwds | new

    def _git(self, args: list[str], dialect: str, state: State, env_assign: bool,
             continues_in_group: bool) -> None:
        bases = state.cwds
        risky = None
        i = 0
        while i < len(args) and args[i].startswith("-"):
            a = args[i]
            name = a.split("=", 1)[0]
            if name in GIT_RISKY_GLOBAL_OPTS:
                risky = name
            if name == "-C":
                value = args[i + 1] if i + 1 < len(args) else ""
                bases = frozenset({None}) if is_dynamic(value) else frozenset(
                    self.policy.resolve(value, b, dialect) for b in bases)
                i += 2
                continue
            i += 2 if (name in GIT_GLOBAL_OPTS_WITH_VALUE and "=" not in a) else 1
        sub = args[i] if i < len(args) else None
        rest = args[i + 1:]
        if env_assign:
            self.flag(ASK, "git invoked with environment variable assignments")
        if risky:
            self.flag(ASK, f"git {risky} can change git's behaviour or make it run other programs")
        if sub is None:
            self.flag(ASK, "git command without a resolvable subcommand")
            return
        if is_dynamic(sub) or continues_in_group and not rest:
            self.flag(ASK, f"git subcommand {sub!r} cannot be resolved")
            return
        if not git_is_read_only(sub, rest):
            self.flag(ASK, f"`git {sub}` is not on the read-only allowlist")
        if sub == "clean":
            self._git_clean(rest, bases, dialect)

    def _git_clean(self, rest: list[str], bases: frozenset, dialect: str) -> None:
        recursive, dry_run, targets, i = False, False, [], 0
        while i < len(rest):
            a = rest[i]
            if a == "--":
                targets.extend(rest[i + 1:])
                break
            if a in ("-e", "--exclude"):
                i += 2
                continue
            if a.startswith("--"):
                dry_run |= a == "--dry-run"
            elif a.startswith("-"):
                dry_run |= "n" in a[1:]
                recursive |= any(ch in self.policy.git_clean_flags for ch in a[1:])
            else:
                targets.append(a)
            i += 1
        if recursive and not dry_run:
            if targets:
                self._check_targets("git clean -d/-x", targets, bases, dialect)
            else:
                self._check_resolved("git clean -d/-x", "the working tree", bases)

    def _delete(self, rule: DeleteRule, name: str, args: list[str], dialect: str, state: State,
                piped_in: bool, stdin_targets: bool, continues_in_group: bool) -> None:
        if continues_in_group:
            self.flag(DENY, f"{rule.name}: `{name}` arguments continue in a sub-expression or brace group "
                            "that cannot be resolved")
            return
        if rule.parser == "posix-rm":
            recursive, targets = parse_posix_rm(args, rule.opts)
        elif rule.parser == "powershell-remove":
            if piped_in:
                self.flag(DENY, f"{rule.name}: `{name}` receives its targets from the pipeline, "
                                "which cannot be verified")
                return
            recursive, targets = parse_ps_remove(args, rule.opts)
        elif rule.parser == "cmd-switch":
            recursive, targets = parse_cmd_switch(args, rule.opts)
        elif rule.parser == "find":
            recursive, targets, opaque = parse_find(args, rule.opts)
            if opaque:
                self.flag(DENY, f"{rule.name}: `{name} -exec {opaque}` runs code that cannot be inspected")
                return
        else:  # "targets": every non-option argument is deleted recursively
            recursive, targets = True, [a for a in args if not a.startswith("-")]
        if not recursive:
            return
        if stdin_targets:
            self.flag(DENY, f"{rule.name}: targets are supplied on stdin and cannot be verified")
            return
        self._check_targets(rule.name, targets, state.cwds, dialect)

    def _check_targets(self, label: str, targets: list[str], bases: frozenset, dialect: str) -> None:
        if not targets:
            self.flag(DENY, f"{label}: no resolvable targets")
            return
        for t in targets:
            if is_dynamic(t):
                self.flag(DENY, f"{label}: target {t!r} uses variables, globs or expansion and cannot be resolved")
                continue
            self._check_resolved(label, repr(t), frozenset(self.policy.resolve(t, b, dialect) for b in bases))

    def _check_resolved(self, label: str, shown: str, paths: frozenset) -> None:
        for p in paths:
            if p is None:
                self.flag(DENY, f"{label}: working directory for {shown} is unknown")
            elif not self.policy.in_whitelist(p):
                if not self.policy.whitelist:
                    self.flag(DENY, f"{label}: {shown} -> {p} is not whitelisted (no folders are whitelisted "
                                    f"in .claude/hooks/rules/{self.policy.rule_file.name})")
                else:
                    self.flag(DENY, f"{label}: {shown} -> {p} is outside the whitelisted folders "
                                    f"in .claude/hooks/rules/{self.policy.rule_file.name}")


def evaluate(tool_name: str, command: str, cwd: str | None, policy: Policy) -> tuple[int, list[str]]:
    dialect = TOOL_DIALECT.get(tool_name)
    if dialect is None:
        return NONE, []
    base = Path(os.path.realpath(cwd)) if cwd else None
    analyzer = Analyzer(policy)
    analyzer.run(command, dialect, frozenset({base}))
    return analyzer.level, analyzer.reasons


# --------------------------------------------------------------------------
# Hook entry point
# --------------------------------------------------------------------------
def emit(level: int, reasons: list[str]) -> None:
    reason = "Repo guard (.claude/hooks/guard.py): " + "; ".join(reasons)
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": DECISION_NAMES[level],
        "permissionDecisionReason": reason,
    }}))


def main() -> int:
    try:
        if sys.version_info < MIN_PYTHON:
            emit(DENY, [f"requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}, running {sys.version.split()[0]}"])
            return 0
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        tool = payload.get("tool_name")
        if tool not in TOOL_DIALECT:
            return 0
        command = (payload.get("tool_input") or {}).get("command")
        if not isinstance(command, str):
            emit(DENY, ["tool input has no command string to inspect"])
            return 0
        policy = load_policy(sys.platform)
        level, reasons = evaluate(tool, command, payload.get("cwd"), policy)
        if level != NONE:
            emit(level, reasons)
        return 0
    except PolicyError as exc:
        emit(DENY, [f"policy error, failing closed: {exc}"])
        return 0
    except Exception as exc:  # noqa: BLE001 - any internal error must fail closed
        emit(DENY, [f"internal error, failing closed: {type(exc).__name__}: {exc}"])
        return 0


if __name__ == "__main__":
    sys.exit(main())
