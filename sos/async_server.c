#include "common.h"

#include <aio.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>
#include <sys/wait.h>

int sockfd;
struct sockaddr_in serv_addr;
struct sockaddr_in cli_addr;
int i;

struct aiocb* new_aiocb(int fd);

static int new_client(int fd)
{
    socklen_t clilen = sizeof(cli_addr);
    int new_fd = accept(fd, (struct sockaddr*)&cli_addr, &clilen);
    if (new_fd == -1)
    {
        error("ERROR accept");
    }
    return new_fd;
}

static void aio_completion_handler(sigval_t sigval)
{
    struct aiocb* cbp = sigval.sival_ptr;
    int fd = cbp->aio_fildes;

    if (fd == sockfd)
    {
        // new client
        int new_fd = new_client(fd);
        struct aiocb* ccbp = new_aiocb(new_fd);

        if (aio_read(ccbp) == -1)
        {
            error("ERROR aio_read");
        }
        write(new_fd, "ACK new", 8);
    }
    else
    {
        int buflen = aio_return(cbp);

        if (buflen > 0)
        {
            char* buf = (void*)cbp->aio_buf;

            buf[buflen] = '\0';
            //printf("server % d %s \n", fd, buf);

            write(fd, "ACK", 4);
            ++i;
        }
        else
        {
            close(fd);
            delete_aiocb(cbp);
            return;
        }

        // next aio_read
        if (aio_read(cbp) == -1)
        {
            error("aio_read");
        }
    }
}

struct aiocb* new_aiocb(int fd)
{
    struct aiocb* cbp = malloc(sizeof(struct aiocb));

    cbp->aio_fildes = fd;
    cbp->aio_offset = 0;
    cbp->aio_buf = malloc(c_buf_size);
    cbp->aio_nbytes = c_buf_size;
    cbp->aio_reqprio = 0;
    cbp->aio_sigevent.sigev_notify = SIGEV_THREAD;
    cbp->aio_sigevent.sigev_signo = 0;
    cbp->aio_sigevent.sigev_value.sival_ptr = cbp;
    cbp->aio_sigevent.sigev_notify_function = aio_completion_handler;
    cbp->aio_sigevent.sigev_notify_attributes = NULL;
    cbp->aio_lio_opcode = LIO_READ;

    return cbp;
}

void delete_aiocb(struct aiocb* cbp)
{
    free((void*)cbp->aio_buf);
    free(cbp);
}

void run_async_server(int portno)
{
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

    struct aiocb* cblist[2];

    cblist[0] = new_aiocb(STDIN_FILENO);
    cblist[1] = new_aiocb(sockfd);

    if (lio_listio(LIO_NOWAIT, cblist, 2, NULL) == -1)
    {
        error("lio_listio");
    }

    while (i < iteration_num)
        ;

    close(sockfd);
}
