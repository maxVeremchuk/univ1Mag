#include "gtest/gtest.h"

extern "C" {
#include "aes.h"
}

namespace {

unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                            {0x68, 0x20, 0x4b, 0x20},
                                            {0x61, 0x6d, 0x75, 0x46},
                                            {0x74, 0x79, 0x6e, 0x75}};

} // anonymous namespace

TEST(AESTest, key_schedule)
{
    unsigned int w[176];
    const unsigned int excpected_w[176] = {
        84,  104, 97,  116, 115, 32,  109, 121, 32,  75,  117, 110, 103, 32,  70,  117, 226, 50,
        252, 241, 145, 18,  145, 136, 177, 89,  228, 230, 214, 121, 162, 147, 86,  8,   32,  7,
        199, 26,  177, 143, 118, 67,  85,  105, 160, 58,  247, 250, 210, 96,  13,  231, 21,  122,
        188, 104, 99,  57,  233, 1,   195, 3,   30,  251, 161, 18,  2,   201, 180, 104, 190, 161,
        215, 81,  87,  160, 20,  82,  73,  91,  177, 41,  59,  51,  5,   65,  133, 146, 210, 16,
        210, 50,  198, 66,  155, 105, 189, 61,  194, 135, 184, 124, 71,  21,  106, 108, 149, 39,
        172, 46,  14,  78,  204, 150, 237, 22,  116, 234, 170, 3,   30,  134, 63,  36,  178, 168,
        49,  106, 142, 81,  239, 33,  250, 187, 69,  34,  228, 61,  122, 6,   86,  149, 75,  108,
        191, 226, 191, 144, 69,  89,  250, 178, 161, 100, 128, 180, 247, 241, 203, 216, 40,  253,
        222, 248, 109, 164, 36,  74,  204, 192, 164, 254, 59,  49,  111, 38};

    key_schedule(w, key);
    for (int i = 0; i < 176; ++i)
    {
        EXPECT_EQ(excpected_w[i], w[i]);
    }
}

TEST(AESTest, add_roundkey)
{
    unsigned int w[176];
    const unsigned int excpected_w[176] = {
        84,  104, 97,  116, 115, 32,  109, 121, 32,  75,  117, 110, 103, 32,  70,  117, 226, 50,
        252, 241, 145, 18,  145, 136, 177, 89,  228, 230, 214, 121, 162, 147, 86,  8,   32,  7,
        199, 26,  177, 143, 118, 67,  85,  105, 160, 58,  247, 250, 210, 96,  13,  231, 21,  122,
        188, 104, 99,  57,  233, 1,   195, 3,   30,  251, 161, 18,  2,   201, 180, 104, 190, 161,
        215, 81,  87,  160, 20,  82,  73,  91,  177, 41,  59,  51,  5,   65,  133, 146, 210, 16,
        210, 50,  198, 66,  155, 105, 189, 61,  194, 135, 184, 124, 71,  21,  106, 108, 149, 39,
        172, 46,  14,  78,  204, 150, 237, 22,  116, 234, 170, 3,   30,  134, 63,  36,  178, 168,
        49,  106, 142, 81,  239, 33,  250, 187, 69,  34,  228, 61,  122, 6,   86,  149, 75,  108,
        191, 226, 191, 144, 69,  89,  250, 178, 161, 100, 128, 180, 247, 241, 203, 216, 40,  253,
        222, 248, 109, 164, 36,  74,  204, 192, 164, 254, 59,  49,  111, 38};

    key_schedule(w, key);
    for (int i = 0; i < 176; ++i)
    {
        EXPECT_EQ(excpected_w[i], w[i]);
    }
}

TEST(AESTest, gfMul)
{
    EXPECT_EQ(20, gfMul(4, 5));
}

TEST(AESTest, sub_byte)
{
    EXPECT_EQ(90, sub_byte(4, 6));
}

TEST(AESTest, shift_rows)
{
    const unsigned int excpected_state[16] = {
        84, 79, 78, 32, 110, 105, 84, 119, 110, 119, 111, 101, 111, 32, 32, 101};

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};

    shift_rows(state);

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(excpected_state[BLOCK_SIZE * i + j], state[i][j]);
        }
    }
}

TEST(AESTest, mix_columns)
{
    const unsigned int excpected_result[16] = {
        126, 105, 44, 164, 43, 28, 75, 126, 157, 139, 84, 43, 164, 154, 31, 157};

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};

    unsigned int result[BLOCK_SIZE][BLOCK_SIZE];

    mix_columns(result, state);

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(excpected_result[BLOCK_SIZE * i + j], result[i][j]);
        }
    }
}

TEST(AESTest, encryption)
{
    const unsigned int excpected_cipher_text[16] = {0x29,
                                                    0x57,
                                                    0x40,
                                                    0x1a,
                                                    0xc3,
                                                    0x14,
                                                    0x22,
                                                    0x2,
                                                    0x50,
                                                    0x20,
                                                    0x99,
                                                    0xd7,
                                                    0x5f,
                                                    0xf6,
                                                    0xb3,
                                                    0x3a};

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};

    unsigned int w[176];

    key_schedule(w, key);
    encryption(state, w);

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(excpected_cipher_text[BLOCK_SIZE * i + j], state[i][j]);
        }
    }
}

TEST(AESTest, inv_sub_byte)
{
    EXPECT_EQ(185, inv_sub_byte(5, 6));
}

TEST(AESTest, inv_shift_rows)
{
    const unsigned int excpected_state[16] = {0x54,
                                              0x4f,
                                              0x4e,
                                              0x20,
                                              0x54,
                                              0x77,
                                              0x6e,
                                              0x69,
                                              0x6e,
                                              0x77,
                                              0x6f,
                                              0x65,
                                              0x20,
                                              0x65,
                                              0x6f,
                                              0x20};

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};

    inv_shift_rows(state);

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(excpected_state[BLOCK_SIZE * i + j], state[i][j]);
        }
    }
}

TEST(AESTest, inv_mix_columns)
{
    const unsigned int excpected_result[16] = {0xdf,
                                               0xcc,
                                               0xd7,
                                               0xae,
                                               0x21,
                                               0x32,
                                               0x0,
                                               0xdf,
                                               0x3c,
                                               0x2e,
                                               0xaf,
                                               0x21,
                                               0xae,
                                               0xb4,
                                               0x54,
                                               0x3cu};

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};

    unsigned int result[BLOCK_SIZE][BLOCK_SIZE];

    inv_mix_columns(result, state);

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(excpected_result[BLOCK_SIZE * i + j], result[i][j]);
        }
    }
}

TEST(AESTest, decryption)
{
    const unsigned int excpected_state[16] = {0x54,
                                              0x4f,
                                              0x4e,
                                              0x20,
                                              0x77,
                                              0x6e,
                                              0x69,
                                              0x54,
                                              0x6f,
                                              0x65,
                                              0x6e,
                                              0x77,
                                              0x20,
                                              0x20,
                                              0x65,
                                              0x6f}; // same as state

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};

    unsigned int w[176];

    key_schedule(w, key);
    encryption(state, w);

    decryption(state, w);

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(excpected_state[BLOCK_SIZE * i + j], state[i][j]);
        }
    }
}
