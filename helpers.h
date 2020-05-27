//
// Some useful routines.
//

#pragma once


using uint = unsigned int;

const float FLT_MAX = std::numeric_limits<float>::max();


inline void crash() __attribute__((noreturn));
inline void crash()
{
#ifndef __clang__
    *(int *) 0 = 0;
#endif
    __builtin_trap();
}

// #ifndef NDEBUG
#undef assert
# define assert(COND) \
    if (!(COND)) { \
        printf("assertion '%s' failed at %s:%d\n", #COND, __FILE__, __LINE__); \
        crash(); \
    }
// #else
// # define assert(COND)
// #endif


void dprint(const char *str, const char *desc, const char *file, int line)
{
    printf("%s:%d %s %s\n", file, line, desc, str);
}

void dprint(int val, const char *desc, const char *file, int line)
{
    printf("%s:%d %s %d\n", file, line, desc, val);
}

void dprint(const json11::Json &json, const char *desc, const char *file, int line)
{
    std::string s;
    json.dump(s);
    printf("%s:%d %s %s\n", file, line, desc, s.c_str());
}

#define DPRINT(VAR) dprint(VAR, #VAR, __FILE__, __LINE__)


int randint(int range)
{
    return std::rand() % range;
}

int randfloat()
{
    return std::rand() / float(RAND_MAX);
}
