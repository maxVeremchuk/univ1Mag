#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRSIZE 256

void swap(unsigned char* a, unsigned char* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

int key_schedule(char* key, unsigned char* S)
{
    int len = strlen(key);
    int j = 0;

    for (int i = 0; i < ARRSIZE; i++)
    {
        S[i] = i;
    }

    for (int i = 0; i < ARRSIZE; i++)
    {
        j = (j + S[i] + key[i % len]) % ARRSIZE;

        swap(&S[i], &S[j]);
    }

    return 0;
}

int cipher(unsigned char* S, char* plaintext, unsigned char* ciphertext)
{
    int i = 0;
    int j = 0;

    for (size_t n = 0, len = strlen(plaintext); n < len; n++)
    {
        i = (i + 1) % ARRSIZE;
        j = (j + S[i]) % ARRSIZE;

        swap(&S[i], &S[j]);
        int cipher_bit = S[(S[i] + S[j]) % ARRSIZE];

        ciphertext[n] = cipher_bit ^ plaintext[n];
    }

    return 0;
}

int RC4(char* key, char* plaintext, unsigned char* ciphertext)
{
    unsigned char S[ARRSIZE];

    key_schedule(key, S);
    cipher(S, plaintext, ciphertext);

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <key> <plaintext>", argv[0]);
        return -1;
    }

    unsigned char* ciphertext = malloc(sizeof(int) * strlen(argv[2]));
    RC4(argv[1], argv[2], ciphertext);

    for (size_t i = 0, len = strlen(argv[2]); i < len; i++)
    {
        printf("%02hhX", ciphertext[i]);
    }
    printf("\n");

    unsigned char* plaintext = malloc(sizeof(int) * strlen(argv[2]));
    RC4(argv[1], ciphertext, plaintext);

    printf("%s\n", plaintext);

    return 0;
}
