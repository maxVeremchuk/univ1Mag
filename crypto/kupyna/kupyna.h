#ifndef SRC_KUPYNA_H_
#define SRC_KUPYNA_H_

#include <stdlib.h>
#include <limits.h>


#define ROWS 8
#define NB_512 8
#define NB_1024 16
#define STATE_BYTE_SIZE_512 (ROWS * NB_512)
#define STATE_BYTE_SIZE_1024 (ROWS * NB_1024)
#define NR_512 10
#define NR_1024 14
#define REDUCTION_POLYNOMIAL 0x011d  /* x^8 + x^4 + x^3 + x^2 + 1 */
#define BLOCK_SIZE 64


#define BITS_IN_WORD 64

#define BITS_IN_BYTE 8


typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef struct {
    uint8_t state[NB_1024][ROWS];
    size_t nbytes;
    size_t data_nbytes;
    uint8_t padding[STATE_BYTE_SIZE_1024 * 2];
    size_t pad_nbytes;
    size_t hash_nbits;
    int columns;
    int rounds;
} kupyna_t;


int KupynaInit(size_t hash_nbits, kupyna_t* ctx);

void KupynaHash(kupyna_t* ctx, uint8_t* data, size_t msg_nbits, uint8_t* hash_code);

#endif
