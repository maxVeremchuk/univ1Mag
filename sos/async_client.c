#include "common.h"

#include <aio.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

void run_async_client(char* host_name, int portno, char* message)
{
    int sockfd, n;
    struct sockaddr_in serv_addr;
    struct sockaddr_in cli_addr;
    struct hostent* server;
    char buffer[c_buf_size];
    static struct aiocb readrq;
    static const struct aiocb* readrqv[2] = {&readrq, NULL};

    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        error("ERROR opening socket");
    }
    server = gethostbyname(host_name);
    if (server == NULL)
    {
        error("ERROR no such host");
    }
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*)server->h_addr_list[0], (char*)&serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(portno);
    bzero(&(serv_addr.sin_zero), 8);

    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof serv_addr))
    {
        error("connecting to server");
    }

    memset(&readrq, 0, sizeof(readrq));
    readrq.aio_fildes = sockfd;
    readrq.aio_buf = buffer;
    readrq.aio_nbytes = c_buf_size;

    write(sockfd, message, (strlen(message)));
    if (aio_read(&readrq))
    {
        error("aio_read");
    }

    int i = 0;
    while (i < iteration_num)
    {
        write(sockfd, message, (strlen(message)));

        aio_suspend(readrqv, 1, NULL);
        int size = aio_return(&readrq);
        if (size >= 0)
        {
            if (aio_read(&readrq))
            {
                error("CLIENT aio_read");
            }
            //printf("cli %s\n", readrq.aio_buf);
        }
        else
        {
            printf("empty aio_read");
        }

        ++i;
    }
    close(sockfd);
}

// int main(int argc, char* argv[])
// {
//     if (argc < 3)
//     {
//         fprintf(stderr, "usage %s hostname port\n", argv[0]);
//         exit(0);
//     }

//     run_inet_client(argv[1], atoi(argv[2]), "d");

//     return 0;
// }
