#ifndef RANDOM_H_
#define RANDOM_H_

#include <chrono>
#include <math.h>

using std::chrono::seconds;
using std::chrono::milliseconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

const unsigned int second = duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count();
const unsigned int millisecond = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
const unsigned int microsecond = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
const unsigned int nanosecond = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

int m_z = second ^ nanosecond;
int m_w = millisecond ^ microsecond;
int state = m_z ^ m_w;

auto start = high_resolution_clock::now();

const unsigned int UIntRandom1()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);

	return (m_z << 16) + m_w;
}

const double DoubleRandom1()	// 0 through 1
{
	return UIntRandom1() * 2.328306435454494e-10;
}

const unsigned int UIntRandom2()
{
	state = (state ^ 2747636419u) * 2654435769u;
	state = (state ^ (state >> 16u)) * 2654435769u;
	state = (state ^ (state >> 16u)) * 2654435769u;

	return state;
}

const double DoubleRandom2()	// 0 through 1
{
	return UIntRandom2() / 4294967295.0;
}

const unsigned int UIntRandom()
{
	return UIntRandom1() ^ UIntRandom2();
}

const int IntRandom()
{
	return UIntRandom();
}

const double DoubleRandom()	// 0 through 1, test |'-'| version
{
	return (DoubleRandom1() + DoubleRandom2()) / 2;
}

const double NormalRandom(double mean, double standardDeviation)
{
	double x, y, radius;
	do
	{
		x = DoubleRandom1() + DoubleRandom2() - 1;
		y = DoubleRandom1() + DoubleRandom2() - 1;

		radius = x * x + y * y;
	} while (radius == 0.0 || radius > 1.0);

	return x * sqrt(-2.0 * log(radius) / radius) * standardDeviation + mean;
}

#endif
