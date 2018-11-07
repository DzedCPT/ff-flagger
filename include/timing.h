
#ifndef TIMING_H
#define TIMING_H

//#define TIME

//#ifdef TIME
#include <chrono>
#include <map>

struct Time {
    double value = 0;
};


#ifdef TIME
#define MARK_TIME(name) name = std::chrono::high_resolution_clock::now();

#define ADD_TIME_SINCE(timer_name, mark) \
	queue.finish(); \
    time_map[#timer_name].value += \
	std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-mark).count();

#endif

#ifndef TIME
std::cout << "No rfi kernel timing" << std::endl;

#define MARK_TIME(name) 

#define ADD_TIME_SINCE(timer_name, mark) 

#endif

#endif

