#ifndef COMPILER_WORKAROUND_H
#define COMPILER_WORKAROUND_H

// Workaround for g++ 15 + OpenMPI 5.0 compatibility issues
#define _GLIBCXX_USE_BUILTIN_TRAIT(X) __has_builtin(X)
#define _GLIBCXX_HAVE_BUILTIN_HAS_UNIQ_OBJ_REP 1
#define _GLIBCXX_HAVE_BUILTIN_IS_AGGREGATE 1
#define _GLIBCXX_HAVE_BUILTIN_IS_SAME 1

#endif