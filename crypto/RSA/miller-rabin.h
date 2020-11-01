#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "bigint.h"

typedef long long ll;

bigint modexp(bigint a, bigint b, bigint n);

bool isprime(bigint n, int rounds);
