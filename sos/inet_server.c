#include "common.h"

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void run_inet_server(int portno)
{
    int sockfd, newsockfd;
    socklen_t clilen;
    char buffer[90];
    struct sockaddr_in serv_addr, cli_addr;
    int err_code;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0)
    {
        error("ERROR opening socket");
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int)) < 0)
    {
        error("setsockopt(SO_REUSEADDR) failed");
    }

    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        error("ERROR on binding");
    }
    listen(sockfd, 5);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen);

    if (newsockfd < 0)
    {
        error("ERROR on accept");
    }

    int i = 0;
    while (i < iteration_num)
    {
        read(newsockfd, buffer, 90);
        write(newsockfd, "ACK", 3);
        //printf("%s\n", buffer);
        ++i;
    }

    close(newsockfd);
    close(sockfd);
}

// int main(int argc, char* argv[])
// {
//     if (argc < 2)
//     {
//         fprintf(stderr, "ERROR, no port provided\n");
//         exit(1);
//     }
//     time_t start, end;
//     start = clock();
//     run_inet_server(atoi(argv[1]));
//      end = clock();
//     printf("Time passed: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);
//     return 0;
// }
