#ifndef KALYNA_DEFS_H
#define KALYNA_DEFS_H


#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <limits.h>

#include "kalyna.h"

#define kBITS_IN_WORD 64

#define kBITS_IN_BYTE 8

#define TRUE 1
#define FALSE 0

/* Block words size. */
#define kNB_128 2

/* Key words size. */
#define kNK_128 2

/* Block bits size. */
#define kBLOCK_128 kNB_128 * kBITS_IN_WORD

/* Block bits size. */
#define kKEY_128 kNK_128 * kBITS_IN_WORD

/* Number of enciphering rounds size depending on key length. */
#define kNR_128 10

#define kREDUCTION_POLYNOMIAL 0x011d  /* x^8 + x^4 + x^3 + x^2 + 1 */

#define INDEX(table, row, col) table[(row) + (col) * sizeof(uint64)]

void SubBytes(kalyna_t* ctx);

void InvSubBytes(kalyna_t* ctx);

void ShiftRows(kalyna_t* ctx);

void InvShiftRows(kalyna_t* ctx);

uint8_t MultiplyGF(uint8_t x, uint8_t y);

void MatrixMultiply(kalyna_t* ctx, uint8_t matrix[8][8]);

void MixColumns(kalyna_t* ctx);

void InvMixColumns(kalyna_t* ctx);

void EncipherRound(kalyna_t* ctx);

void DecipherRound(kalyna_t* ctx);

void AddRoundKey(int round, kalyna_t* ctx);

void SubRoundKey(int round, kalyna_t* ctx);

void AddRoundKeyExpand(uint64* value, kalyna_t* ctx);

void XorRoundKey(int round, kalyna_t* ctx);

void XorRoundKeyExpand(uint64* value, kalyna_t* ctx);

void Rotate(size_t state_size, uint64* state_value);

void ShiftLeft(size_t state_size, uint64* state_value);

void RotateLeft(size_t state_size, uint64* state_value);

void KeyExpandKt(uint64* key, kalyna_t* ctx, uint64* kt);

void KeyExpandEven(uint64* key, uint64* kt, kalyna_t* ctx);

void KeyExpandOdd(kalyna_t* ctx);

uint8_t* WordsToBytes(size_t length, uint64* words);

uint64* BytesToWords(size_t length, uint8_t* bytes);

uint64 ReverseWord(uint64 word);

int IsBigEndian();

void PrintState(size_t length, uint64* state);

#endif  /* KALYNA_DEFS_H */

