#include <stdint.h>
#include <stdio.h>

#define R(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define quarter(a, b, c, d) \
    b ^= R(d + a, 7);       \
    c ^= R(a + b, 9);       \
    d ^= R(b + c, 13);      \
    a ^= R(c + d, 18);

#define LE(p) ((p)[0] | ((p)[1] << 8) | ((p)[2] << 16) | ((p)[3] << 24))

void salsa20_words(uint32_t* out, uint32_t in[16])
{
    uint32_t x[4][4];
    int i;
    for (i = 0; i < 16; ++i) x[i / 4][i % 4] = in[i];
    for (i = 0; i < 10; ++i)
    {
        quarter(x[0][0], x[1][0], x[2][0], x[3][0]);
        quarter(x[1][1], x[2][1], x[3][1], x[0][1]);
        quarter(x[2][2], x[3][2], x[0][2], x[1][2]);
        quarter(x[3][3], x[0][3], x[1][3], x[2][3]);

        quarter(x[0][0], x[0][1], x[0][2], x[0][3]);
        quarter(x[1][1], x[1][2], x[1][3], x[1][0]);
        quarter(x[2][2], x[2][3], x[2][0], x[2][1]);
        quarter(x[3][3], x[3][0], x[3][1], x[3][2]);
    }
    for (i = 0; i < 16; ++i)
    {
        out[i] = x[i / 4][i % 4] + in[i];
    }
}

void salsa20_block(uint8_t* out, uint8_t key[32], uint64_t nonce, uint64_t index)
{
    static const char c[16] = "expand 32-byte k";

    // clang-format off
    uint32_t in[16] = {LE(c),            LE(key),    LE(key+4),        LE(key+8),
	                   LE(key+12),       LE(c+4),    nonce&0xffffffff, nonce>>32,
	                   index&0xffffffff, index>>32,  LE(c+8),          LE(key+16),
	                   LE(key+20),       LE(key+24), LE(key+28),       LE(c+12)};
    // clang-format on

    uint32_t wordout[16];
    salsa20_words(wordout, in);
    int i;
    for (i = 0; i < 64; ++i)
    {
        out[i] = 0xff & (wordout[i / 4] >> (8 * (i % 4)));
    }
}

void salsa20(uint8_t* message, uint64_t mlen, uint8_t key[32], uint64_t nonce)
{
    int i;
    uint8_t block[64];
    for (i = 0; i < mlen; i++)
    {
        if (i % 64 == 0)
        {
            salsa20_block(block, key, nonce, i / 64);
        }
        message[i] ^= block[i % 64];
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <key> <plaintext>", argv[0]);
        return -1;
    }

    uint8_t key[32] = {0};
    uint64_t nonce = 0;
    uint8_t plaintext[64] = {0};

    for (size_t i = 0; i < 32; ++i)
    {
        key[i] = (uint8_t)argv[2][i];
    }

    for (size_t i = 0; i < 64; ++i)
    {
        plaintext[i] = (uint8_t)argv[2][i];
    }

    salsa20(plaintext, sizeof(plaintext), key, nonce);
    int i;

    for (i = 0; i < sizeof(plaintext); ++i)
    {
        printf("%02X", plaintext[i]);
    }
    printf("\n");

    salsa20(plaintext, sizeof(plaintext), key, nonce);
    printf("%s", plaintext);

    return 0;
}
