#pragma once

#include <iostream>

//#define PROFILE

#ifdef PROFILE
#include <chrono>
#define PROFILE_ITERATIONS 1000
#endif