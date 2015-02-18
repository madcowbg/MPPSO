#ifndef __UTILITIES_CPP
#define __UTILITIES_CPP

#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////
// Common host and device function 
///////////////////////////////////////////////////////////////////////////////
///ceil(a / b)
extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

///floor(a / b)
extern "C" int iDivDown(int a, int b){
    return a / b;
}

///Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

///Align a to nearest lower multiple of b
extern "C" int iAlignDown(int a, int b){
    return a - a % b;
}

#endif

