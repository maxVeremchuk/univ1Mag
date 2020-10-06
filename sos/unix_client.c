#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

void run_unix_client(char* socket_path, char* message)
{
    struct sockaddr_un serv_addr;
    char buffer[90];
    int sockfd, n;

    while (!is_socket_file_created)
        ;


    if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
    {
        error("Creating socket");
    }

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sun_family = AF_UNIX;
    strncpy(serv_addr.sun_path, socket_path, sizeof(serv_addr.sun_path) - 1);

    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        error("Connecting");
    }
    int i = 0;
    while (i < iteration_num)
    {
        bzero(buffer,90);
        write(sockfd, message, strlen(message));
        //read(sockfd, buffer, strlen(buffer));
        //printf("ccc %d  %s\n", i, buffer);
        ++i;
    }
    close(sockfd);
}
