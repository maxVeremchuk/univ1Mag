#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

void run_unix_server(char* socket_path)
{
    int sockfd, newsockfd, servlen, n;
    socklen_t clilen;
    struct sockaddr_un cli_addr, serv_addr;
    char buffer[c_buf_size];

    if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
    {
        error("creating socket");
    }

    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sun_family = AF_UNIX;
    snprintf(serv_addr.sun_path, 108, socket_path);
    strcpy(serv_addr.sun_path, socket_path);
    servlen = strlen(serv_addr.sun_path) + sizeof(serv_addr.sun_family);
    if (bind(sockfd, (struct sockaddr*)&serv_addr, servlen) < 0)
    {
        error("binding socket");
    }
    is_socket_file_created++;

    listen(sockfd, 5);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen);
    if (newsockfd < 0)
    {
        error("accepting");
    }

    int i = 0;
    while (i < iteration_num)
    {
        bzero(buffer, c_buf_size);
        read(newsockfd, buffer, c_buf_size);
        write(newsockfd, "ACK", 3);
        ++i;
    }
    close(newsockfd);
    close(sockfd);
}
