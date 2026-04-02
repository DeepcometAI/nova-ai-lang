cmake_minimum_required(VERSION 3.20)
project(nova_parser C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(nova_parser STATIC
    src/parser.c
    src/ast.c
)

target_include_directories(nova_parser PUBLIC
    include
    ../lexer/include
)

target_link_libraries(nova_parser PUBLIC nova_lexer)

# Shared library wrapper for Python/Rust FFI.
add_library(nova_parser_ffi SHARED
    src/parser_ffi.c
)

target_include_directories(nova_parser_ffi PUBLIC
    include
    ../lexer/include
)

target_link_libraries(nova_parser_ffi PRIVATE nova_parser nova_lexer)

