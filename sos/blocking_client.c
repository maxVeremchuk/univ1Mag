#include "common.h"

#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void run_blocking_client(char* host_name, int portno, char *message)
{
    int sockfd, n;
    struct sockaddr_in serv_addr;
    struct hostent* server;

    char buffer[256];
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        error("ERROR opening socket");
    }
    server = gethostbyname(host_name);
    if (server == NULL)
    {
        fprintf(stderr, "ERROR, no such host\n");
        exit(0);
    }
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*)server->h_addr_list[0], (char*)&serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        error("ERROR connecting");
    }

    n = write(sockfd, message, strlen(message));
    if (n < 0)
    {
        error("ERROR writing to socket");
    }
    bzero(buffer, 256);
    n = read(sockfd, buffer, 255);
    if (n < 0)
    {
        error("ERROR reading from socket");
    }
    printf("%s\n", buffer);
    close(sockfd);
}

// int main(int argc, char* argv[])
// {
//     if (argc < 3)
//     {
//         fprintf(stderr, "usage %s hostname port\n", argv[0]);
//         exit(0);
//     }

//     run_blocking_client(argv[1], atoi(argv[2]), "d");

//     return 0;
// }
