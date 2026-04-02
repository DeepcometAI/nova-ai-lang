#ifndef NOVA_PARSER_FFI_H
#define NOVA_PARSER_FFI_H

/*
 * Minimal C FFI surface for the NOVA C lexer+parser.
 *
 * Purpose: allow Python (ctypes) / Rust (bindgen) callers to parse NOVA source
 * using the existing C parser without first wiring the whole compiler.
 *
 * The API returns heap-allocated UTF-8 strings. The caller MUST free them by
 * calling `nova_ffi_free`.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parse the given source and return:
 *  - out_dump: a textual AST dump (similar to nova_ast_print output)
 *  - out_errors: any lexer/parser errors, or empty string on success
 *
 * Returns 0 on success (no errors), non-zero on failure.
 */
int nova_parse_dump(const char *src, size_t src_len, const char *filename,
                    char **out_dump, char **out_errors);

/* Free a string returned by nova_parse_dump. Safe on NULL. */
void nova_ffi_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_PARSER_FFI_H */

