#include "datautils.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>

int main()
{
    char buffer[strlen(c_message)];
    int p[2], pid, nbytes;

    time_t start, end;

    start = clock();

    if (pipe(p) < 0)
    {
        exit(1);
    }

    if ((pid = fork()) > 0)
    {
        for (int i = 0; i < 100000; i++)
        {
            write(p[1], c_message, strlen(c_message));
        }
        close(p[1]);
    }

    else
    {
        while ((nbytes = read(p[0], buffer, strlen(c_message))) > 0)
        {
            // printf("% s\n", buffer);
        }
        if (nbytes != 0)
        {
            exit(1);
        }
    }

    end = clock();

    printf("Time passed pipes: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    return 0;
}
