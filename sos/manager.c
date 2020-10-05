#include <pthread.h>
#include <stdio.h>
#include <time.h>

#include "blocking_client.c"
#include "blocking_server.c"
#include "unix_client.c"
#include "unix_server.c"

const char* socket_path = "socket";
const int portno = 51717;
const char* message = "message message";
const char* hostname = "localhost";

void run_all_servers()
{
    printf("run_all_servers\n");
    run_unix_server(socket_path);
    printf("after run_all_servers\n");
    // pthread_barrier_wait (&barrier);

    // run_blocking_server(portno);
}

void run_all_clients()
{
    printf("run_all_clients\n");
    sleep(1);

    run_unix_client(socket_path, message);
    printf("after run_all_clients\n");
    // pthread_barrier_wait (&barrier);

    // run_blocking_client(args->hostname, args->portno, args->message);

}

void run_unix_sockets()
{
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, run_all_servers, NULL);
    pthread_create(&thread2, NULL, run_all_clients, NULL);

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
    // pthread_t thread1, thread2;

   // pthread_barrier_init(&barrier, NULL, 3);


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

    end = clock();
    printf("Time passed: %ld\n", (end - start));

    if (remove(socket_path))
    {
        error("File deletion");
    }

    return 0;
}
