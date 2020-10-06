#ifndef COMMON_H
#define COMMON_H

#define _POSIX_C_SOURCE 200112L

#include <pthread.h>

pthread_barrier_t  barrier;

int is_socket_file_created = 0;

const int iteration_num = 100000;

void error(const char* msg)
{
    perror(msg);
    exit(1);
}

#endif // COMMON_H
