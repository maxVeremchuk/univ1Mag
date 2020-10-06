#include <pthread.h>
#include <stdio.h>
#include <time.h>

#include "inet_client.c"
#include "inet_server.c"
#include "unix_client.c"
#include "unix_server.c"

const char* socket_path = "socket";
const int portno = 51717;
const char* message = "message messagemessage messagemessage messagemessage messagemessage messagemessage message";
const char* hostname = "localhost";

///unix
void run_unix_server_manager()
{
    run_unix_server(socket_path);
}

void run_unix_client_manager()
{
    run_unix_client(socket_path, message);
}

void run_unix_sockets() //0.208613  20.641946 s     5: 104.149935 s  10: 210.859216
{
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, run_unix_server_manager, NULL);
    pthread_create(&thread2, NULL, run_unix_client_manager, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
}

///inet
void run_inet_server_manager()
{
    run_inet_server(portno);
}

void run_inet_client_manager()
{
    run_inet_client(hostname, portno, message);
}

void run_inet_sockets() //0.1 : 0.177633  1: 1.629690 10: 16.13388  50:82.72321 s 100: 163.954484
{
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, run_inet_server_manager, NULL);
    pthread_create(&thread2, NULL, run_inet_client_manager, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
}

// void run_blocking_client_manager(void *arguments)
// {
//     struct arg_struct *args = arguments;
//     run_blocking_client(args->hostname, args->portno, args->message);
// }

// void run_unix_client_manager(void *arguments)
// {
//     struct arg_struct *args = arguments;
//     run_unix_client(args->portno, args->message);
// }

int main(int argc, char* argv[])
{
    printf("_____MANAGER_____\n");

    time_t start, end;
    start = clock();

    // pthread_create(&thread1, NULL, run_all_servers, portno);
    // pthread_create(&thread2, NULL, run_all_clients, (void*)&args);

    // //     pthread_barrier_wait (&barrier);
    // //     printf("____BARRIER_____\n");
    // // sleep(3);
    // pthread_join(thread1, NULL);
    // pthread_join(thread2, NULL);

    run_unix_sockets();
    //run_inet_sockets();


    end = clock();
    printf("Time passed: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    if (remove(socket_path))
    {
        error("File deletion");
    }

    return 0;
}
