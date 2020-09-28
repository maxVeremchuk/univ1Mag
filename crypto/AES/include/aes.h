#ifndef AES_H
#define AES_H

#define BYTES 16

#define BLOCK_SIZE 4

void key_schedule(unsigned int[], unsigned int[][BLOCK_SIZE]);
void add_roundkey(
    unsigned int[][BLOCK_SIZE],
    unsigned int[][BLOCK_SIZE],
    unsigned int[][BLOCK_SIZE]);

unsigned int sub_byte(unsigned int, unsigned int);
void shift_rows(unsigned int[][BLOCK_SIZE]);
void mix_columns(unsigned int[][BLOCK_SIZE], unsigned int[][BLOCK_SIZE]);
void encryption(unsigned int[][BLOCK_SIZE], unsigned int[]);

unsigned int inv_sub_byte(unsigned int, unsigned int);
void inv_shift_rows(unsigned int[][BLOCK_SIZE]);
void inv_mix_columns(unsigned int[][BLOCK_SIZE], unsigned int[][BLOCK_SIZE]);
void decryption(unsigned int[][BLOCK_SIZE], unsigned int[]);

int hexCharToDec(char);
void get2Bytes(unsigned int, unsigned int*, unsigned int*);
void getRoundKey(unsigned int[], unsigned int[][BLOCK_SIZE], int);
unsigned int gfMul(unsigned int, unsigned int);
void convertStringToBlock(char[], unsigned int[][BLOCK_SIZE]);
void convertBlockToString(unsigned int[][BLOCK_SIZE], char[]);

void printArray(unsigned int[][BLOCK_SIZE]);

void test();

#endif // AES_H
