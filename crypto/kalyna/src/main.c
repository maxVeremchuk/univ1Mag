#include <memory.h>
#include <stdio.h>
#include <time.h>

#include "kalyna.h"
#include "transformations.h"

void print(int data_size, uint64 data[]);

int main(int argc, char** argv)
{
    time_t start, end;
    start = clock();

    FILE* stream;

    kalyna_t* ctx22_e = kalyna_init();
    uint64 pt22_e[2] = {0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL};
    uint64 ct22_e[2];
    uint64 key22_e[2] = {0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
    // uint64 expect22_e[2] = {0x20ac9b777d1cbf81ULL, 0x06add2b439eac9e1ULL};

    for (int counter = 0; counter < 100; ++counter)
    {
        printf("iteration %d\n\n\n", counter);
        if ((stream = fopen("test.txt", "rb")) == NULL)
        {
            printf("Can't open ");
            exit(1);
        }
        char buffer[16];
        unsigned long long int block_1;
        unsigned long long int block_2;

        int read = 0;
        while ((read = fread(buffer, 1, 16, stream)) > 0)
        {
            block_1 = 0;
            block_2 = 0;
            char block_1[(8 * 2) + 1];
            char block_2[(8 * 2) + 1];

            char* ptr_1 = &block_1[0];
            for (int i = 7; i >= 0; --i)
            {
                ptr_1 += sprintf(ptr_1, "%02X", buffer[i]);
            }

            char* ptr_2 = &block_2[0];
            for (int i = 15; i >= 8; --i)
            {
                ptr_2 += sprintf(ptr_2, "%02X", buffer[i]);
            }
            char * pEnd;

            pt22_e[0] = strtoull(block_1, pEnd, 16);
            pt22_e[1] = strtoull(block_2, pEnd, 16);
            kalyna_key_expand(key22_e, ctx22_e);
            kalyna_encipher(pt22_e, ctx22_e, ct22_e);
        }
    }

    end = clock();
    printf("%f", ((double)(end - start)) / CLOCKS_PER_SEC);
    fclose(stream);

    return 0;
}

void print(int data_size, uint64 data[])
{
    int i;
    uint8_t* tmp = (uint8_t*)data;
    for (i = 0; i < data_size * 8; i++)
    {
        if (!(i % 16)) printf("    ");
        printf("%02X", (unsigned int)tmp[i]);
        if (!((i + 1) % 16)) printf("\n");
    };
    printf("\n");
};
