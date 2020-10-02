#include <stdio.h>
#include <string.h>
#include <time.h>

#include "aes.h"

int main()
{
    time_t start, end;
    start = clock();

    FILE* stream; // = fopen("test.txt", "r");
    if ((stream = fopen("test.txt", "rb")) == NULL)
    {
        printf("Can't open ");
        exit(1);
    }

    unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
    unsigned int w[176];
    key_schedule(w, key);
    for (int counter = 0; counter < 20; ++counter)
    {
        printf("iteration %d\n\n\n", counter);
        unsigned int state[BLOCK_SIZE][BLOCK_SIZE];
        char buffer[BLOCK_SIZE * BLOCK_SIZE];

        int count = 0;
        int read = 0;
        while((read = fread(buffer, 1, 16, stream)) > 0)
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
            count++;
            if(count > 1000)
            {
                count = 0;
                printf("encryption...");
            }
        }
    }

    end = clock();
    printf("%f", ((double)(end - start)) / CLOCKS_PER_SEC); //196
    fclose(stream);
    //test();

    return 0;
}
