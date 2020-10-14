#include <pthread.h>
#include <stdio.h>
#include <time.h>

#include "async_client.c"
#include "async_server.c"
#include "inet_client.c"
#include "inet_server.c"
#include "nonblock_client.c"
#include "nonblock_server.c"
#include "unix_client.c"
#include "unix_server.c"

const char* socket_path = "socket";
const int portno = 51717;
const char* message =
    "message messagemessage messagemessage messagemessage messagemessage messagemessage message";
const char* hostname = "localhost";

/// unix
void run_unix_server_manager()
{
    run_unix_server(socket_path);
}

void run_unix_client_manager()
{
    run_unix_client(socket_path, message);
}

void run_unix_sockets()
{
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, run_unix_server_manager, NULL);
    pthread_create(&thread2, NULL, run_unix_client_manager, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
}

/// inet
void run_inet_server_manager()
{
    run_inet_server(portno);
}

void run_inet_client_manager()
{
    run_inet_client(hostname, portno, message);
}

void run_inet_sockets()
{
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, run_inet_server_manager, NULL);
    pthread_create(&thread2, NULL, run_inet_client_manager, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
}

/// nonblock
void run_nonblock_server_manager()
{
    run_nonblock_server(portno);
}

void run_nonblock_client_manager()
{
    run_nonblock_client(hostname, portno, message);
}

void run_nonblock_sockets()
{
    pthread_t thread1, thread2, thread3, thread4;

    pthread_create(&thread1, NULL, run_nonblock_server_manager, NULL);
    pthread_create(&thread2, NULL, run_nonblock_client_manager, NULL);
    pthread_create(&thread3, NULL, run_nonblock_client_manager, NULL);
    pthread_create(&thread4, NULL, run_nonblock_client_manager, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);
}

/// async
void run_async_server_manager()
{
    run_async_server(portno);
}

void run_async_client_manager()
{
    sleep(1);
    run_async_client(hostname, portno, message);
}

void run_async_sockets()
{
    pthread_t thread1, thread2, thread3, thread4;

    pthread_create(&thread1, NULL, run_async_server_manager, NULL);
    pthread_create(&thread2, NULL, run_async_client_manager, NULL);
    //pthread_create(&thread3, NULL, run_async_client_manager, NULL);
    //pthread_create(&thread4, NULL, run_async_client_manager, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    //pthread_join(thread3, NULL);
    //pthread_join(thread4, NULL);
}

int main(int argc, char* argv[])
{
    printf("_____MANAGER_____\n");

    time_t start, end;

    // start = clock();
    // run_unix_sockets();
    // end = clock();
    // printf("Time passed unix: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // start = clock();
    // run_inet_sockets();
    // end = clock();
    // printf("Time passed inet: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // start = clock();
    // run_nonblock_sockets();
    // end = clock();
    // printf("Time passed non block 3 clients: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    run_async_sockets();
    end = clock();
    printf("Time passed async 3 clients: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    if (remove(socket_path))
    {
        error("File deletion");
    }

    return 0;
}
