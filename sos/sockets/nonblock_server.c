#include "common.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void run_nonblock_server(int portno)
{
    int sockfd, last_fd, new_fd;
    int sin_size;
    socklen_t clilen;
    char buffer[c_buf_size];
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

    if ((new_fd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen)) < 0)
    {
        perror("no available accept first");
    }
    last_fd = new_fd;

    fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFL, 0) | O_NONBLOCK);
    fcntl(new_fd, F_SETFL, O_NONBLOCK);

    int last = 0;
    int sockets[3];
    sockets[last++] = new_fd;
    int n;
    int i = 0;
    while (1)
    {
        if ((new_fd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen)) > 0)
        {
            fcntl(new_fd, F_SETFL, O_NONBLOCK);
            sockets[last++] = new_fd;
        }

        for (int socket_num = 0; socket_num < last; socket_num++)
        {
            bzero(buffer, c_buf_size);
            n = recv(sockets[socket_num], buffer, sizeof(buffer), 0);

            if (n > 0)
            {
                if (send(sockets[socket_num], "ACK\n", 4, 0) < 0)
                {
                    error("ERROR on send");
                }
                ++i;
            }
        }

        if (i >= 3 * iteration_num)
        {
            break;
        }
    }

   for (int socket_num = 0; socket_num < last; socket_num++)
    {
        close(sockets[socket_num]);
    }
    close(sockfd);
}
