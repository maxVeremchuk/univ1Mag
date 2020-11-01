#include "miller-rabin.h"

bigint modexp(bigint a, bigint b, bigint n)
{
    bigint c = 1;
    while (!(b <= 1))
    {
        if (b % 2 == 0)
        {
            a *= a;
            a = a % n;
            b = b / 2;
        }
        else
        {
            c = c * a;
            c = c % n;
            b = b - 1;
        }
    }
    return (a * c) % n;
}

bool isprime(bigint n, int rounds)
{
    if (n == 2) return true;
    if (n == 1 || n % 2 == 0) return false;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g1(seed);
    std::uniform_int_distribution<ll> distribution(0, base - 1);
    std::vector<bigint> bases(rounds);
    for (int c = 0; c < rounds; c++)
    {
        std::string to_bigint_str = "";

        for (int i = 0; i < n.a.size(); ++i)
        {
            ll generated = distribution(g1);
            to_bigint_str.append(std::to_string(generated));
        }
        bigint temp(to_bigint_str);
        if (n <= temp)
        {
            c--;
        }
        else
        {
            bases[c] = temp;
        }
    }
    bigint d, x, b;
    ll k;
    ll s = 0;
    d = n - 1;
    while (d % 2 == 0)
    {
        s = s + 1;
        d = d / 2;
    }
    for (int c = 0; c < rounds; c++)
    {
        b = bases[c];
        if (n <= b) continue;
        if (!(gcd(b, n) == 1))
        {
            if (b == n)
                continue;
            else
                return false;
        }
        x = modexp(b, d, n);
        if ((x == 1) || (x == n - 1)) continue;
        for (k = 1; k < s; k++)
        {
            x = (x * x) % n;
            if (x == n - 1) break;
            if (x == 1) return false;
        }
        if (k == s) return false;
        if (s == 1 && !(x == n - 1)) return false;
    }
    return true;
}
