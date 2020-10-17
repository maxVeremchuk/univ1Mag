#include <stdio.h>
#include <string.h>
#include <time.h>

#include "aes.h"

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
    const unsigned int input_vector[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
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

void CFB()
{
    FILE* in_stream;

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];
    const unsigned int input_vector[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                               {0x68, 0x20, 0x4b, 0x20},
                                                               {0x61, 0x6d, 0x75, 0x46},
                                                               {0x74, 0x79, 0x6e, 0x75}};
    const int SMALL_BLOCK_SIZE = BLOCK_SIZE - 2;
    unsigned int state[SMALL_BLOCK_SIZE][SMALL_BLOCK_SIZE];
    char buffer[BLOCK_SIZE * BLOCK_SIZE];
    unsigned int xor_vector[BLOCK_SIZE][BLOCK_SIZE];
    unsigned int new_xor_vector[BLOCK_SIZE][BLOCK_SIZE];
    unsigned int input_xor_vector[BLOCK_SIZE][BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            xor_vector[i][j] = input_vector[i][j];
        }
    }

    if ((in_stream = fopen("test.txt", "rb")) == NULL)
    {
        printf("Can't open ");
        exit(1);
    }

    int read = 0;
    int i = 0;
    while ((read = fread(buffer, 1, 4, in_stream)) > 0)
    {
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                input_xor_vector[i][j] = xor_vector[i][j];
            }
        }

        key_schedule(w, key);
        encryption(xor_vector, w);

        int i_xor = 0;
        int j_xor = 0;

        for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
            {
                state[j][i] = (int)buffer[i * SMALL_BLOCK_SIZE + j] ^ xor_vector[j_xor][i_xor];
                j_xor++;
                if (j_xor == BLOCK_SIZE)
                {
                    j_xor = 0;
                    i_xor++;
                }
            }
        }

        for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
            {
                printf("%02hhx ", state[j][i]);
            }
        }

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                xor_vector[j][i] = input_xor_vector[j][i];
            }
        }
        int position = BLOCK_SIZE * BLOCK_SIZE - SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE;
        i_xor = position / BLOCK_SIZE;
        j_xor = position % BLOCK_SIZE;
        for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
            {
                xor_vector[i_xor][j_xor] = state[j][i];
                j_xor++;
                if (j_xor == BLOCK_SIZE)
                {
                    j_xor = 0;
                    i_xor++;
                }
            }
        }

        // for (int i = 0; i < BLOCK_SIZE; ++i)
        // {
        //     for (int j = 0; j < BLOCK_SIZE; ++j)
        //     {
        //         printf("%02hhx ", xor_vector[j][i]);
        //     }
        //     printf("\n");
        // }
    }
}

void OFB()
{
    FILE* in_stream;

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];
    const unsigned int input_vector[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                               {0x68, 0x20, 0x4b, 0x20},
                                                               {0x61, 0x6d, 0x75, 0x46},
                                                               {0x74, 0x79, 0x6e, 0x75}};

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

    if ((in_stream = fopen("test.txt", "rb")) == NULL)
    {
        printf("Can't open ");
        exit(1);
    }

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

        key_schedule(w, key);
        encryption(xor_vector, w);

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                state[i][j] = xor_vector[i][j] ^ state[i][j];
            }
        }

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                printf("%02hhx ", state[j][i]);
            }
        }
    }
}

void CTR()
{
    FILE* in_stream;

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];
    const unsigned int input_vector[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                               {0x68, 0x20, 0x4b, 0x20},
                                                               {0x61, 0x6d, 0x75, 0x46},
                                                               {0x74, 0x79, 0x6e, 0x75}};

    unsigned int state[BLOCK_SIZE][BLOCK_SIZE];
    char buffer[BLOCK_SIZE * BLOCK_SIZE];
    unsigned int xor_vector[BLOCK_SIZE][BLOCK_SIZE];

    if ((in_stream = fopen("test.txt", "rb")) == NULL)
    {
        printf("Can't open ");
        exit(1);
    }

    int read = 0;
    while ((read = fread(buffer, 1, 16, in_stream)) > 0)
    {
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                xor_vector[i][j] = input_vector[i][j];
            }
        }

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                state[j][i] = (int)buffer[i * BLOCK_SIZE + j];
            }
        }

        key_schedule(w, key);
        encryption(xor_vector, w);

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                state[i][j] = xor_vector[i][j] ^ state[i][j];
            }
        }

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                printf("%02hhx ", state[j][i]);
            }
        }
    }
}

int main()
{
    // time_measurement();
    printf("\n-----------ECB-----------\n");
    ECB();
    printf("\n-----------CBC-----------\n");
    CBC();
    printf("\n-----------CFB-----------\n");
    CFB();
    printf("\n-----------OFB-----------\n");
    OFB();
    printf("\n-----------CTR-----------\n");
    CTR();
    printf("\n-----------END-----------\n");
    return 0;
}
