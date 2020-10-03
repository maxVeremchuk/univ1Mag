#include "tables.h"
#include "transformations.h"

kalyna_t* kalyna_init()
{
    int i;
    kalyna_t* ctx = (kalyna_t*)malloc(sizeof(kalyna_t));

    ctx->nb = kBLOCK_128 / kBITS_IN_WORD;

    ctx->nk = kKEY_128 / kBITS_IN_WORD;
    ctx->nr = kNR_128;

    ctx->state = (uint64*)calloc(ctx->nb, sizeof(uint64));
    if (ctx->state == NULL)
    {
        perror("Could not allocate memory for cipher state.");
    }

    ctx->round_keys = (uint64**)calloc(ctx->nr + 1, sizeof(uint64**));
    if (ctx->round_keys == NULL)
    {
        perror("Could not allocate memory for cipher round keys.");
    }

    for (i = 0; i < ctx->nr + 1; ++i)
    {
        ctx->round_keys[i] = (uint64*)calloc(ctx->nb, sizeof(uint64));
        if (ctx->round_keys[i] == NULL)
        {
            perror("Could not allocate memory for cipher round keys.");
        }
    }
    return ctx;
}

int kalyna_delete(kalyna_t* ctx)
{
    int i;
    free(ctx->state);
    for (i = 0; i < ctx->nr + 1; ++i)
    {
        free(ctx->round_keys[i]);
    }
    free(ctx->round_keys);
    free(ctx);
    ctx = NULL;
    return 0;
}

void SubBytes(kalyna_t* ctx)
{
    int i;
    uint64* s = ctx->state; /* For shorter expressions. */
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = sboxes_enc[0][s[i] & 0x00000000000000FFULL] |
                        ((uint64)sboxes_enc[1][(s[i] & 0x000000000000FF00ULL) >> 8] << 8) |
                        ((uint64)sboxes_enc[2][(s[i] & 0x0000000000FF0000ULL) >> 16] << 16) |
                        ((uint64)sboxes_enc[3][(s[i] & 0x00000000FF000000ULL) >> 24] << 24) |
                        ((uint64)sboxes_enc[0][(s[i] & 0x000000FF00000000ULL) >> 32] << 32) |
                        ((uint64)sboxes_enc[1][(s[i] & 0x0000FF0000000000ULL) >> 40] << 40) |
                        ((uint64)sboxes_enc[2][(s[i] & 0x00FF000000000000ULL) >> 48] << 48) |
                        ((uint64)sboxes_enc[3][(s[i] & 0xFF00000000000000ULL) >> 56] << 56);
    }
}

void InvSubBytes(kalyna_t* ctx)
{
    int i;
    uint64* s = ctx->state; /* For shorter expressions. */
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = sboxes_dec[0][s[i] & 0x00000000000000FFULL] |
                        ((uint64)sboxes_dec[1][(s[i] & 0x000000000000FF00ULL) >> 8] << 8) |
                        ((uint64)sboxes_dec[2][(s[i] & 0x0000000000FF0000ULL) >> 16] << 16) |
                        ((uint64)sboxes_dec[3][(s[i] & 0x00000000FF000000ULL) >> 24] << 24) |
                        ((uint64)sboxes_dec[0][(s[i] & 0x000000FF00000000ULL) >> 32] << 32) |
                        ((uint64)sboxes_dec[1][(s[i] & 0x0000FF0000000000ULL) >> 40] << 40) |
                        ((uint64)sboxes_dec[2][(s[i] & 0x00FF000000000000ULL) >> 48] << 48) |
                        ((uint64)sboxes_dec[3][(s[i] & 0xFF00000000000000ULL) >> 56] << 56);
    }
}

void ShiftRows(kalyna_t* ctx)
{
    int row, col;
    int shift = -1;

    uint8_t* state = WordsToBytes(ctx->nb, ctx->state);
    uint8_t* nstate = (uint8_t*)malloc(ctx->nb * sizeof(uint64));

    for (row = 0; row < sizeof(uint64); ++row)
    {
        if (row % (sizeof(uint64) / ctx->nb) == 0) shift += 1;
        for (col = 0; col < ctx->nb; ++col)
        {
            INDEX(nstate, row, (col + shift) % ctx->nb) = INDEX(state, row, col);
        }
    }

    ctx->state = BytesToWords(ctx->nb * sizeof(uint64), nstate);
    free(state);
}

void InvShiftRows(kalyna_t* ctx)
{
    int row, col;
    int shift = -1;

    uint8_t* state = WordsToBytes(ctx->nb, ctx->state);
    uint8_t* nstate = (uint8_t*)malloc(ctx->nb * sizeof(uint64));

    for (row = 0; row < sizeof(uint64); ++row)
    {
        if (row % (sizeof(uint64) / ctx->nb) == 0) shift += 1;
        for (col = 0; col < ctx->nb; ++col)
        {
            INDEX(nstate, row, col) = INDEX(state, row, (col + shift) % ctx->nb);
        }
    }

    ctx->state = BytesToWords(ctx->nb * sizeof(uint64), nstate);
    free(state);
}

uint8_t MultiplyGF(uint8_t x, uint8_t y)
{
    int i;
    uint8_t r = 0;
    uint8_t hbit = 0;
    for (i = 0; i < kBITS_IN_BYTE; ++i)
    {
        if ((y & 0x1) == 1) r ^= x;
        hbit = x & 0x80;
        x <<= 1;
        if (hbit == 0x80) x ^= kREDUCTION_POLYNOMIAL;
        y >>= 1;
    }
    return r;
}

void MatrixMultiply(kalyna_t* ctx, uint8_t matrix[8][8])
{
    int col, row, b;
    uint8_t product;
    uint64 result;
    uint8_t* state = WordsToBytes(ctx->nb, ctx->state);

    for (col = 0; col < ctx->nb; ++col)
    {
        result = 0;
        for (row = sizeof(uint64) - 1; row >= 0; --row)
        {
            product = 0;
            for (b = sizeof(uint64) - 1; b >= 0; --b)
            {
                product ^= MultiplyGF(INDEX(state, b, col), matrix[row][b]);
            }
            result |= (uint64)product << (row * sizeof(uint64));
        }
        ctx->state[col] = result;
    }
}

void MixColumns(kalyna_t* ctx)
{
    MatrixMultiply(ctx, mds_matrix);
}

void InvMixColumns(kalyna_t* ctx)
{
    MatrixMultiply(ctx, mds_inv_matrix);
}

void EncipherRound(kalyna_t* ctx)
{
    SubBytes(ctx);
    ShiftRows(ctx);
    MixColumns(ctx);
}

void DecipherRound(kalyna_t* ctx)
{
    InvMixColumns(ctx);
    InvShiftRows(ctx);
    InvSubBytes(ctx);
}

void AddRoundKey(int round, kalyna_t* ctx)
{
    int i;
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = ctx->state[i] + ctx->round_keys[round][i];
    }
}

void SubRoundKey(int round, kalyna_t* ctx)
{
    int i;
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = ctx->state[i] - ctx->round_keys[round][i];
    }
}

void AddRoundKeyExpand(uint64* value, kalyna_t* ctx)
{
    int i;
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = ctx->state[i] + value[i];
    }
}

void XorRoundKey(int round, kalyna_t* ctx)
{
    int i;
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = ctx->state[i] ^ ctx->round_keys[round][i];
    }
}

void XorRoundKeyExpand(uint64* value, kalyna_t* ctx)
{
    int i;
    for (i = 0; i < ctx->nb; ++i)
    {
        ctx->state[i] = ctx->state[i] ^ value[i];
    }
}

void Rotate(size_t state_size, uint64* state_value)
{
    int i;
    uint64 temp = state_value[0];
    for (i = 1; i < state_size; ++i)
    {
        state_value[i - 1] = state_value[i];
    }
    state_value[state_size - 1] = temp;
}

void ShiftLeft(size_t state_size, uint64* state_value)
{
    int i;
    for (i = 0; i < state_size; ++i)
    {
        state_value[i] <<= 1;
    }
}

void RotateLeft(size_t state_size, uint64* state_value)
{
    size_t rotate_bytes = 2 * state_size + 3;
    size_t bytes_num = state_size * (kBITS_IN_WORD / kBITS_IN_BYTE);

    uint8_t* bytes = WordsToBytes(state_size, state_value);
    uint8_t* buffer = (uint8_t*)malloc(rotate_bytes);

    memcpy(buffer, bytes, rotate_bytes);
    memmove(bytes, bytes + rotate_bytes, bytes_num - rotate_bytes);
    memcpy(bytes + bytes_num - rotate_bytes, buffer, rotate_bytes);

    state_value = BytesToWords(bytes_num, bytes);

    free(buffer);
}

void KeyExpandKt(uint64* key, kalyna_t* ctx, uint64* kt)
{
    uint64* k0 = (uint64*)malloc(ctx->nb * sizeof(uint64));
    uint64* k1 = (uint64*)malloc(ctx->nb * sizeof(uint64));

    memset(ctx->state, 0, ctx->nb * sizeof(uint64));
    ctx->state[0] += ctx->nb + ctx->nk + 1;

    if (ctx->nb == ctx->nk)
    {
        memcpy(k0, key, ctx->nb * sizeof(uint64));
        memcpy(k1, key, ctx->nb * sizeof(uint64));
    }
    else
    {
        memcpy(k0, key, ctx->nb * sizeof(uint64));
        memcpy(k1, key + ctx->nb, ctx->nb * sizeof(uint64));
    }

    AddRoundKeyExpand(k0, ctx);
    EncipherRound(ctx);
    XorRoundKeyExpand(k1, ctx);
    EncipherRound(ctx);
    AddRoundKeyExpand(k0, ctx);
    EncipherRound(ctx);
    memcpy(kt, ctx->state, ctx->nb * sizeof(uint64));

    free(k0);
    free(k1);
}

void KeyExpandEven(uint64* key, uint64* kt, kalyna_t* ctx)
{
    int i;
    uint64* initial_data = (uint64*)malloc(ctx->nk * sizeof(uint64));
    uint64* kt_round = (uint64*)malloc(ctx->nb * sizeof(uint64));
    uint64* tmv = (uint64*)malloc(ctx->nb * sizeof(uint64));
    size_t round = 0;

    memcpy(initial_data, key, ctx->nk * sizeof(uint64));
    for (i = 0; i < ctx->nb; ++i)
    {
        tmv[i] = 0x0001000100010001;
    }

    while (TRUE)
    {
        memcpy(ctx->state, kt, ctx->nb * sizeof(uint64));
        AddRoundKeyExpand(tmv, ctx);
        memcpy(kt_round, ctx->state, ctx->nb * sizeof(uint64));

        memcpy(ctx->state, initial_data, ctx->nb * sizeof(uint64));

        AddRoundKeyExpand(kt_round, ctx);
        EncipherRound(ctx);
        XorRoundKeyExpand(kt_round, ctx);
        EncipherRound(ctx);
        AddRoundKeyExpand(kt_round, ctx);

        memcpy(ctx->round_keys[round], ctx->state, ctx->nb * sizeof(uint64));

        if (ctx->nr == round) break;

        if (ctx->nk != ctx->nb)
        {
            round += 2;

            ShiftLeft(ctx->nb, tmv);

            memcpy(ctx->state, kt, ctx->nb * sizeof(uint64));
            AddRoundKeyExpand(tmv, ctx);
            memcpy(kt_round, ctx->state, ctx->nb * sizeof(uint64));

            memcpy(ctx->state, initial_data + ctx->nb, ctx->nb * sizeof(uint64));

            AddRoundKeyExpand(kt_round, ctx);
            EncipherRound(ctx);
            XorRoundKeyExpand(kt_round, ctx);
            EncipherRound(ctx);
            AddRoundKeyExpand(kt_round, ctx);

            memcpy(ctx->round_keys[round], ctx->state, ctx->nb * sizeof(uint64));

            if (ctx->nr == round) break;
        }
        round += 2;
        ShiftLeft(ctx->nb, tmv);
        Rotate(ctx->nk, initial_data);
    }

    free(initial_data);
    free(kt_round);
    free(tmv);
}

void KeyExpandOdd(kalyna_t* ctx)
{
    int i;
    for (i = 1; i < ctx->nr; i += 2)
    {
        memcpy(ctx->round_keys[i], ctx->round_keys[i - 1], ctx->nb * sizeof(uint64));
        RotateLeft(ctx->nb, ctx->round_keys[i]);
    }
}

void kalyna_key_expand(uint64* key, kalyna_t* ctx)
{
    uint64* kt = (uint64*)malloc(ctx->nb * sizeof(uint64));
    KeyExpandKt(key, ctx, kt);
    KeyExpandEven(key, kt, ctx);
    KeyExpandOdd(ctx);
    free(kt);
}

void kalyna_encipher(uint64* plaintext, kalyna_t* ctx, uint64* ciphertext)
{
    int round = 0;
    memcpy(ctx->state, plaintext, ctx->nb * sizeof(uint64));

    AddRoundKey(round, ctx);
    for (round = 1; round < ctx->nr; ++round)
    {
        EncipherRound(ctx);
        XorRoundKey(round, ctx);
    }
    EncipherRound(ctx);
    AddRoundKey(ctx->nr, ctx);

    memcpy(ciphertext, ctx->state, ctx->nb * sizeof(uint64));
}

void kalyna_decipher(uint64* ciphertext, kalyna_t* ctx, uint64* plaintext)
{
    int round = ctx->nr;
    memcpy(ctx->state, ciphertext, ctx->nb * sizeof(uint64));

    SubRoundKey(round, ctx);
    for (round = ctx->nr - 1; round > 0; --round)
    {
        DecipherRound(ctx);
        XorRoundKey(round, ctx);
    }
    DecipherRound(ctx);
    SubRoundKey(0, ctx);

    memcpy(plaintext, ctx->state, ctx->nb * sizeof(uint64));
}

uint8_t* WordsToBytes(size_t length, uint64* words)
{
    int i;
    uint8_t* bytes;
    if (IsBigEndian())
    {
        for (i = 0; i < length; ++i)
        {
            words[i] = ReverseWord(words[i]);
        }
    }
    bytes = (uint8_t*)words;
    return bytes;
}

uint64* BytesToWords(size_t length, uint8_t* bytes)
{
    int i;
    uint64* words = (uint64*)bytes;
    if (IsBigEndian())
    {
        for (i = 0; i < length; ++i)
        {
            words[i] = ReverseWord(words[i]);
        }
    }
    return words;
}

uint64 ReverseWord(uint64 word)
{
    int i;
    uint64 reversed = 0;
    uint8_t* src = (uint8_t*)&word;
    uint8_t* dst = (uint8_t*)&reversed;

    for (i = 0; i < sizeof(uint64); ++i)
    {
        dst[i] = src[sizeof(uint64) - i];
    }
    return reversed;
}

int IsBigEndian()
{
    unsigned int num = 1;
    /* Check the least significant byte value to determine endianness */
    return (*((uint8_t*)&num) == 0);
}

void PrintState(size_t length, uint64* state)
{
    int i;
    for (i = length - 1; i >= 0; --i)
    {
        printf("0x%x, ", state[i]);
    }
    printf("\n");
}
