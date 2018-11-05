
#ifndef TIMING_H
#define TIMING_H

#define TIME

#ifdef TIME
#include <chrono>

//#define INIT_TIMING std::chrono::high_resolution_clock::time_point timer;
#define INIT_MARK(name) std::chrono::high_resolution_clock::time_point name;

#define MARK_TIME(name) name = std::chrono::high_resolution_clock::now();

#define INIT_TIMER(name) std::chrono::duration<double> name;

#define ADD_TIME_SINCE_MARK(timer, mark) timer += std::chrono::high_resolution_clock::now() - mark;

#define PRINT_TIMER(name) std::cout << #name << " took " << name.count() << " seconds" << std::endl;

#endif

#ifndef TIME

#define INIT_MARK(name);

#define MARK_TIME(name);

#define INIT_TIMER(name);

#define ADD_TIME_SINCE_MARK(timer, mark);

#define PRINT_TIMER(name);

#endif

#endif

