#include <stdio.h>
#include <string.h>
#include <time.h>

#include "aes.h"

const unsigned int input_vector[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                           {0x68, 0x20, 0x4b, 0x20},
                                                           {0x61, 0x6d, 0x75, 0x46},
                                                           {0x74, 0x79, 0x6e, 0x75}};

void time_measurement()
{
    time_t start, end;
    start = clock();

    FILE* stream;

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];
    key_schedule(w, key);
    for (int counter = 0; counter < 100; ++counter)
    {
        printf("iteration %d\n\n\n", counter);
        if ((stream = fopen("test.txt", "rb")) == NULL)
        {
            printf("Can't open ");
            exit(1);
        }
        unsigned int state[BLOCK_SIZE][BLOCK_SIZE];
        char buffer[BLOCK_SIZE * BLOCK_SIZE];

        int read = 0;
        while ((read = fread(buffer, 1, 16, stream)) > 0)
        {
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                for (int j = 0; j < BLOCK_SIZE; ++j)
                {
                    state[i][j] = buffer[i * BLOCK_SIZE + j];
                }
            }

            key_schedule(w, key);
            encryption(state, w);
        }
    }

    end = clock();
    printf("%f", ((double)(end - start)) / CLOCKS_PER_SEC); // 4103.13 s
    fclose(stream);
}

void ECB()
{
    FILE* in_stream;

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];


//54 73 20 67 68 20 4b 20 61 6d 75 46 74 79 6e 75


//54 4F 4E 20 77 6E 69 54 6F 65 6E 77 20 20 65 6F


    if ((in_stream = fopen("test.txt", "rb")) == NULL)
    {
        printf("Can't open ");
        exit(1);
    }
    unsigned int state[BLOCK_SIZE][BLOCK_SIZE];
    char buffer[BLOCK_SIZE * BLOCK_SIZE];
key_schedule(w, key);
    int read = 0;
    while ((read = fread(buffer, 1, 16, in_stream)) > 0)
    {
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                state[j][i] = (int)buffer[i * BLOCK_SIZE + j];
            }

        }
        encryption(state, w);
        //decryption(state, w);
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                printf("%02hhx ", state[j][i]);
            }
        }


    }
}

void CBC()
{
    FILE* in_stream;

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];

    if ((in_stream = fopen("test.txt", "rb")) == NULL)
    {
        printf("Can't open ");
        exit(1);
    }
    unsigned int state[BLOCK_SIZE][BLOCK_SIZE];
    char buffer[BLOCK_SIZE * BLOCK_SIZE];
    unsigned int xor_vector[BLOCK_SIZE][BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            xor_vector[i][j] = input_vector[i][j];
        }
    }

    int read = 0;
    int i = 0;
    while ((read = fread(buffer, 1, 16, in_stream)) > 0)
    {
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                state[j][i] = (int)buffer[i * BLOCK_SIZE + j] ^ xor_vector[j][i];
            }
        }

        key_schedule(w, key);
        encryption(state, w);

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                printf("%02hhx ", state[j][i]);
            }
        }
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                xor_vector[i][j] = state[i][j];
            }
        }
    }
}
//13751ACC5A78EB5FCBC3EC3D2FCC638ECD1A15A0EE0CDFE7EBDF9AE8EF8020DE8068FDFAED6CAB24AB1737F7514307D6FC09090F5A070763A0613C8132651AE39DE8CB67891A8533E2D51C9F11905454
//13751acc5a78eb5fcbc3ec3d2fcc638ecd1a15a0ee0cdfe7ebdf9ae8ef8020de8068fdfaed6cab24ab1737f7514307d6fc09090f5a070763a0613c8132651ae3fcfd76d78eb6d10201b32ee67d6f6130

int main()
{
    // time_measurement();
    printf("\n-----------ECB-----------\n");
    ECB();
    printf("\n-----------CBC-----------\n");
    CBC();
    printf("\n-----------END-----------\n");
    return 0;
}
