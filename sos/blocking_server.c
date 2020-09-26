#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void error_server(const char* msg)
{
    perror(msg);
    exit(1);
}

void run_blocking_server(int portno)
{

    int sockfd, newsockfd;
    socklen_t clilen;
    char buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int err_code;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0)
    {
        error_server("ERROR opening socket");
    }
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        error_server("ERROR on binding");
    }
    listen(sockfd, 5);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen);

    if (newsockfd < 0)
    {
        error_server("ERROR on accept");
    }
    bzero(buffer, 256);
    err_code = read(newsockfd, buffer, 255);

    if (err_code < 0)
    {
        error_server("ERROR reading from socket");
    }
    printf("Message: %s\n", buffer);

    err_code = write(newsockfd, "ACK true", 8);

    if (err_code < 0)
    {
        error_server("ERROR writing to socket");
    }
    close(newsockfd);
    close(sockfd);
     printf("close servers");
}

// int main(int argc, char* argv[])
// {
//     if (argc < 2)
//     {
//         fprintf(stderr, "ERROR, no port provided\n");
//         exit(1);
//     }

//     run_blocking_server(atoi(argv[1]));

//     return 0;
// }
