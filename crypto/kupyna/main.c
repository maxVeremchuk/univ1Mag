#include <memory.h>
#include <stdio.h>
#include <time.h>
#include "kupyna.h"

int main(int argc, char** argv)
{
    kupyna_t ctx;
    uint8_t hash_code[BLOCK_SIZE];

    uint8_t test[1000] = {0x00,
                          0x01,
                          0x02,
                          0x03,
                          0x04,
                          0x05,
                          0x06,
                          0x07,
                          0x08,
                          0x09,
                          0x0A,
                          0x0B,
                          0x0C,
                          0x0D,
                          0x0E,
                          0x0F};

    uint8_t hash_256[BLOCK_SIZE] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    printf("=============\nKupyna-256\n\n");

    unsigned char init_text[] = {"Hello world!"};
    unsigned char tries_char[1000];
    long long int tries = 0;

    KupynaInit(256, &ctx);
    time_t start, end;
    start = clock();

    while (1)
    {
        unsigned char text[1000];
        strcpy(text, init_text);
        sprintf(tries_char, "%lld", tries);
        strcat(text, tries_char);
        KupynaHash(&ctx, text, 512, hash_code);

        if (memcmp(hash_code, hash_256, 3) == 0)
        {
            printf("PoW");
            break;
        }
       //if (tries % 1000000 == 0)
        {
            printf("%s \n", text);
        }

        tries++;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            printf("%x", hash_code[i]);
        }
        printf("\n", text);
    }

    end = clock();
    printf("%f  ", ((double)(end - start)) / CLOCKS_PER_SEC);

    return 0;
}
