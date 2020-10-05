#ifndef COMMON_H
#define COMMON_H

#define _POSIX_C_SOURCE 200112L

#include <pthread.h>

pthread_barrier_t  barrier;

void error(const char* msg)
{
    perror(msg);
    exit(1);
}

#endif // COMMON_H
