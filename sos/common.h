#ifndef COMMON_H
#define COMMON_H

#include <pthread.h>


int is_socket_file_created = 0;

const int iteration_num = 1;
const int c_buf_size = 90;

void error(const char* msg)
{
    perror(msg);
    exit(1);
}

#endif // COMMON_H
