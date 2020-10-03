#include "gtest/gtest.h"

extern "C" {
#include "kalyna.h"
#include "tables.h"
#include "transformations.h"
}

TEST(AESTest, SubBytes)
{
    kalyna_t* ctx22_e = kalyna_init();
    SubBytes(ctx22_e);
    const unsigned int excpected_state[] = {1754517160, 1754517160};
    for (int i = ctx22_e->nb - 1; i >= 0; --i)
    {
        EXPECT_EQ(excpected_state[i], (int)ctx22_e->state[i]);
    }
}

TEST(AESTest, InvSubBytes)
{
    kalyna_t* ctx22_e = kalyna_init();
    uint64 key22_e[2] = {0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
    kalyna_key_expand(key22_e, ctx22_e);
    InvSubBytes(ctx22_e);
    const unsigned int excpected_state[] = {1987719642, 465803454};
    for (int i = ctx22_e->nb - 1; i >= 0; --i)
    {
        EXPECT_EQ(excpected_state[i], (int)ctx22_e->state[i]);
    }
}

TEST(AESTest, round_keys)
{
    kalyna_t* ctx22_e = kalyna_init();
    uint64 key22_e[2] = {0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
    kalyna_key_expand(key22_e, ctx22_e);
    const unsigned int excpected_round_keys[] = {
        0x6b5e5016, 0xdc775b86, 0x775b86e6, 0x5e5016f4, 0x6e87707e, 0xaa0aa8a,
        0xa0aa8a76, 0x87707e42, 0xc5d4ce45, 0x8276723e, 0x76723ef5, 0xd4ce45fe,
        0x22ee778c, 0x32665f51, 0x665f5162, 0xee778cb1, 0xe272980a, 0x209a87aa,
        0x9a87aab8, 0x72980ad8, 0xa8b12657, 0xd5f30bf6};
    for (int i = 0; i < ctx22_e->nr + 1; ++i)
    {
        for (int j = 0; j < ctx22_e->nb; ++j)
        {
            EXPECT_EQ(excpected_round_keys[i * (ctx22_e->nb) + j], (int)ctx22_e->round_keys[i][j]);
        }
    }
}

TEST(AESTest, kalyna_encipher)
{
    kalyna_t* ctx22_e = kalyna_init();
    uint64 pt22_e[2] = {0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL};
    uint64 ct22_e[2];
    uint64 key22_e[2] = {0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
    uint64 expect22_e[2] = {0x20ac9b777d1cbf81ULL, 0x06add2b439eac9e1ULL};
    kalyna_key_expand(key22_e, ctx22_e);
    kalyna_encipher(pt22_e, ctx22_e, ct22_e);
    for (int i = 0; i < ctx22_e->nb; ++i)
    {
        EXPECT_EQ(expect22_e[i], ctx22_e->state[i]);
    }
}

TEST(AESTest, kalyna_decipher)
{
    uint64 ct22_d[2] = {0x18191a1b1c1d1e1fULL, 0x1011121314151617ULL};
    uint64 pt22_d[2];
    uint64 key22_d[2] = {0x08090a0b0c0d0e0fULL, 0x0001020304050607ULL};
    uint64 expect22_d[2] = {0x84c70c472bef9172ULL, 0xd7da733930c2096fULL};
    kalyna_t* ctx22_d = kalyna_init();
    kalyna_key_expand(key22_d, ctx22_d);
    kalyna_decipher(ct22_d, ctx22_d, pt22_d);
    for (int i = 0; i < ctx22_d->nb; ++i)
    {
        EXPECT_EQ(expect22_d[i], ctx22_d->state[i]);
    }
}
