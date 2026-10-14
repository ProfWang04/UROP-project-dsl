# L1 grammar (toy / v1 freeze)

> Status: describes the **toy sanity** grammar. Phase 1 main needs a harder L1 freeze (k=6+, strict A∘B, cross-statement deps) — see plan. Keep as v1 reference.

Space-separated tokens. Vocab ≈ 40 tokens (`<pad> <bos> <eos> <unk>` + digits + ops + parens + APIs + vars `a`–`h`).

## Statements

```text
program := stmt+
stmt    := set VAR expr
         | print expr
         | inc VAR | dec VAR | clr VAR          # extras by api_k
         | add VAR VAR | mul VAR VAR | mov VAR VAR
```

`api_k` selects how many APIs are enabled (min 2 = `set`+`print`; then `inc`, `dec`, `add`, `mul`, `clr`, `mov`).

## Expressions (shared with L0)

```text
expr  := term (("+" | "-") term)*
term  := factor ("*" factor)*
factor := DIGIT | VAR | "(" expr ")"
```

Single-digit integers `0`–`9`. `*` binds tighter than `+`/`-`.

## Semantics

- Env: all vars init to `0`.
- `set x e` — `x := eval(e)`
- `print e` — append `eval(e)` to outputs
- `inc`/`dec`/`clr`/`add`/`mul`/`mov` — as named
- **Accept** = parse + execute without error (Phase 1 primary whole-program metric)

## Hold-out

Train never sees programs containing both held-out APIs `A` and `B`; `test_holdout` forces `A∘B`.
