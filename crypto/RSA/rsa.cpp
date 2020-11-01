#include "rsa.h"

bigint ExtEuclid(bigint a, bigint b)
{
    bigint x = 0, y = 1, u = 1, v = 0, gcd = b, m, n, q, r;
    while (!(a == 0))
    {
        q = gcd / a;
        r = gcd % a;
        m = x - u * q;
        n = y - v * q;
        gcd = a;
        a = r;
        x = u;
        y = v;
        u = m;
        v = n;
    }
    return y;
}

// bigint rsa_modExp(bigint b, bigint e, bigint m)
// {
//     bigint x(1), y(b);
//     bigint _1(1);
//     // static int i = 0;
//     while (!(e <= 0))
//     {
//         if (e % 2 == _1)
//         {
//             x = (x * y) % m;
//         }
//         y = (y * y) % m;
//         if (e % 2 == _1)
//         {
//             e -= 1;
//             e -= e / 2;
//         }
//         else
//         {
//             e -= e / 2;
//         }
//         // std::cout << "i ----- " << i << std::endl;
//         // std::cout << "y ----- " << y << std::endl;
//         // std::cout << "x ----- " << x << std::endl;
//         // std::cout << "e ----- " << e << std::endl;
//         // std::cout << "m ----- " << m << std::endl;
//         // ++i;
//     }
//     return x % m;
// }

void rsa_gen_keys(
    struct public_key* pub,
    struct private_key* priv,
    std::string p_prime,
    std::string q_prime)
{
    bigint p(p_prime);
    bigint q(q_prime);

    bigint e(exp_constant);

    bigint max((p * q));
    bigint phi_max((p - 1) * (q - 1));

    bigint d(ExtEuclid(phi_max, e));
    while (d < 0)
    {
        d = d + phi_max;
    }
    pub->modulus = max;
    pub->exponent = e;

    priv->modulus = max;
    priv->modulus_p = p;
    priv->modulus_q = q;
    priv->exponent = d;
}

bigint rsa_encrypt(const bigint message, const struct public_key* pub)
{
    bigint encrypted(modexp(message, pub->exponent, pub->modulus));
    return encrypted;
}

bigint rsa_decrypt(const bigint message, const struct private_key* priv)
{
    bigint d_p(priv->exponent % (priv->modulus_p - 1));
    bigint d_q(priv->exponent % (priv->modulus_q - 1));
    bigint decrypted_p(modexp(message, d_p, priv->modulus_p));
    bigint decrypted_q(modexp(message, d_q, priv->modulus_q));
    bigint qInv(ExtEuclid(priv->modulus_q, priv->modulus_p));
    bigint h((qInv * (decrypted_p - decrypted_q)) % priv->modulus_p);
    bigint decrypted(decrypted_q + (h * priv->modulus_q));
    return decrypted;
}

bigint miller_rabin_generator(std::string s, int digiit_amount)
{
    for (int count = 0; count < 1000; ++count)
    {
        s = "";
        s.append(std::to_string(rand() % 9 + 1));
        for (int i = 0; i < digiit_amount; i++)
        {
            s.append(std::to_string(rand() % 900 + 100));
        }
        s.append(std::to_string(rand() % 90 + 10));
        std::cout << s << std::endl;
        bigint N(s);
        if (isprime(N, 5))
        {
            return N;
        }
    }
}

void miller_rabin_check(std::string s)
{
    bigint N(s);
    if (isprime(N, 5))
    {
        std::cout << "YES" << std::endl;
    }
    else
    {
        std::cout << "NO" << std::endl;
    }
}

int main()
{
    init_prime_digit();
    struct public_key* pub = new public_key(0, 0);
    struct private_key* priv = new private_key(0, 0);
    bigint message("1234567");
    time_t start, end;
    for (int test = 0; test < prime_digits.size(); ++test)
    {
        std::cout << "p:: " << prime_digits[test].first << std::endl;
        std::cout << "q:: " << prime_digits[test].second << std::endl;
        // std::cout << "-----Miller Rabin Check-----" << std::endl;
        // miller_rabin_check(prime_digits[test].first);
        // miller_rabin_check(prime_digits[test].second);
        // std::cout << "-----Miller Rabin Check End-----" << std::endl;

        std::cout << "message:: " << message << std::endl;

        std::cout << "-----Start RSA-----" << std::endl;
        start = clock();
        rsa_gen_keys(pub, priv, prime_digits[test].first, prime_digits[test].second);
        bigint encrypted(rsa_encrypt(message, pub));
        std::cout << "encrypted:: " << encrypted << std::endl;
        bigint decrypted(rsa_decrypt(encrypted, priv));
        std::cout << "decrypted:: " << decrypted << std::endl;

        end = clock();
        std::cout << "-----End RSA-----" << std::endl;
        std::cout << "Time " <<  ((double)(end - start)) / CLOCKS_PER_SEC << std::endl;
    }
    return 0;
}
