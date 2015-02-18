

#ifndef __UTILITIES_H
#define __UTILITIES_H


///Fast mul on G8x / G9x / G100
#define IMUL(a, b) __mul24(a, b)

///////////////////////////////////////////////////////////////////////////////
// Common host and device function 
///////////////////////////////////////////////////////////////////////////////
///ceil(a / b)
extern "C" int iDivUp(int a, int b);

///floor(a / b)
extern "C" int iDivDown(int a, int b);

///Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b);

///Align a to nearest lower multiple of b
extern "C" int iAlignDown(int a, int b);


#endif

