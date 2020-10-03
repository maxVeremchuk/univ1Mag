#ifndef KALYNA_H
#define KALYNA_H


#include <stdlib.h>
#include <string.h>

typedef unsigned char uint8_t;
typedef unsigned long long uint64;

typedef struct {
    size_t nb;
    size_t nk;
    size_t nr;
    uint64* state;
    uint64** round_keys;
} kalyna_t;


kalyna_t* kalyna_init();

int kalyna_delete(kalyna_t* ctx);

void kalyna_key_expand(uint64* key, kalyna_t* ctx);

void kalyna_encipher(uint64* plaintext, kalyna_t* ctx, uint64* ciphertext);

void kalyna_decipher(uint64* ciphertext, kalyna_t* ctx, uint64* plaintext);

#endif  /* KALYNA_H */

