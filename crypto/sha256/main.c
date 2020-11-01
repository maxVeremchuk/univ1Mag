#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sha256.h"

int proof_of_work(int partial_colision_num)
{
    BYTE zero_hash[SHA256_BLOCK_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    BYTE buf[SHA256_BLOCK_SIZE];
    SHA256_CTX ctx;
    long long int tries = 0;

    BYTE init_text[] = {"Hello world!"};
    BYTE tries_char[1000];

    while (1)
    {
        BYTE text[1000];
        strcpy(text, init_text);
        sprintf(tries_char, "%lld", tries);
        strcat(text, tries_char);
        sha256_init(&ctx);
        sha256_update(&ctx, text, strlen(text));
        sha256_final(&ctx, buf);
        if (!memcmp(zero_hash, buf, partial_colision_num))
        {
            return 1;
        }

        if (tries % 1000000 == 0)
        {
            printf("%s \n", text);
        }
        // for (int i = 0; i < SHA256_BLOCK_SIZE; ++i)
        // {
        //     printf("%x ", buf[i]);
        // }
        // printf("\n");
        tries++;
    }
}

int main()
{
    time_t start, end;
    start = clock();

    proof_of_work(3);

    end = clock();
    printf("%f  ", ((double)(end - start)) / CLOCKS_PER_SEC); // 20.757338
    return (0);
}
