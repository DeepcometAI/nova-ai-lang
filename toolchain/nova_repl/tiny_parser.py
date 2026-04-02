"""
Tiny NOVA parser (prototype).

Goal: support a minimal end-to-end slice while the real compiler pipeline is
being wired:

  mission main() → Void {
    transmit("Hello, universe!")
  }

This file intentionally implements a *very small* tokenizer + parser and
returns a JSON-serializable AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class TinyParseError(ValueError):
    pass


@dataclass(frozen=True)
class Tok:
    kind: str
    text: str
    pos: int


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_part(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(src: str) -> List[Tok]:
    toks: List[Tok] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            i += 1
            continue

        # symbols
        if src.startswith("->", i):
            toks.append(Tok("ARROW", "->", i))
            i += 2
            continue
        if ch == "→":
            toks.append(Tok("ARROW", "→", i))
            i += 1
            continue
        if ch in "(){},":
            toks.append(Tok(ch, ch, i))
            i += 1
            continue

        # string literal (double quotes, no escapes for now)
        if ch == '"':
            j = i + 1
            while j < n and src[j] != '"':
                # minimal escape support for \" and \\ (enough for demos)
                if src[j] == "\\" and j + 1 < n:
                    j += 2
                else:
                    j += 1
            if j >= n or src[j] != '"':
                raise TinyParseError(f"unterminated string literal at {i}")
            toks.append(Tok("STRING", src[i : j + 1], i))
            i = j + 1
            continue

        # identifier / keyword
        if _is_ident_start(ch):
            j = i + 1
            while j < n and _is_ident_part(src[j]):
                j += 1
            text = src[i:j]
            kind = "KW" if text in {"mission", "Void"} else "IDENT"
            toks.append(Tok(kind, text, i))
            i = j
            continue

        raise TinyParseError(f"unexpected character {ch!r} at {i}")
    toks.append(Tok("EOF", "", n))
    return toks


class Parser:
    def __init__(self, toks: List[Tok]):
        self.toks = toks
        self.k = 0

    def _cur(self) -> Tok:
        return self.toks[self.k]

    def _eat(self, kind: str) -> Tok:
        t = self._cur()
        if t.kind != kind:
            raise TinyParseError(f"expected {kind}, got {t.kind} ({t.text!r}) at {t.pos}")
        self.k += 1
        return t

    def _eat_text(self, kind: str, text: str) -> Tok:
        t = self._cur()
        if t.kind != kind or t.text != text:
            raise TinyParseError(f"expected {text!r}, got {t.text!r} at {t.pos}")
        self.k += 1
        return t

    def parse_program(self) -> Dict[str, Any]:
        mission = self.parse_mission()
        self._eat("EOF")
        return {"kind": "Program", "items": [mission]}

    def parse_mission(self) -> Dict[str, Any]:
        self._eat_text("KW", "mission")
        name = self._eat("IDENT").text
        self._eat("(")
        self._eat(")")
        self._eat("ARROW")
        # return type
        rt = self._cur()
        if rt.kind == "KW" and rt.text == "Void":
            self._eat("KW")
            ret_type = {"kind": "TypeIdent", "name": "Void"}
        else:
            raise TinyParseError(f"only Void return type supported in tiny parser (at {rt.pos})")
        self._eat("{")
        body = [self.parse_stmt()]
        self._eat("}")
        return {
            "kind": "MissionDecl",
            "name": name,
            "params": [],
            "return_type": ret_type,
            "body": body,
        }

    def parse_stmt(self) -> Dict[str, Any]:
        # Only: transmit("...")
        callee = self._eat("IDENT").text
        if callee != "transmit":
            raise TinyParseError("tiny parser only supports transmit(...) statement")
        self._eat("(")
        s = self._eat("STRING").text
        self._eat(")")
        return {
            "kind": "ExprStmt",
            "expr": {
                "kind": "CallExpr",
                "callee": {"kind": "Ident", "name": callee},
                "args": [{"kind": "StringLit", "value": _unquote_string(s)}],
            },
        }


def _unquote_string(tok_text: str) -> str:
    # tok_text includes the surrounding quotes
    assert tok_text.startswith('"') and tok_text.endswith('"')
    inner = tok_text[1:-1]
    # minimal escape handling
    out = []
    i = 0
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt in {'"', "\\"}:
                out.append(nxt)
                i += 2
                continue
        out.append(inner[i])
        i += 1
    return "".join(out)


def parse_tiny(src: str) -> Dict[str, Any]:
    toks = tokenize(src)
    return Parser(toks).parse_program()

