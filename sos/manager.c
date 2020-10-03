#include <time.h>
#include <pthread.h>

#include "blocking_client.c"
#include "blocking_server.c"

struct arg_struct {
    int portno;
    char* message;
    char* hostname;
};

void run_blocking_client_manager(void *arguments)
{
    struct arg_struct *args = arguments;
    run_blocking_client(args->hostname, args->portno, args->message);
}

int main(int argc, char* argv[])
{
    const int portno = 51717;
    char* message = "message message";
    pthread_t thread1, thread2;
    struct arg_struct args;
    args.portno = portno;
    args.message = message;
    args.hostname = "localhost";

    time_t start, end;
    start = clock();

    pthread_create( &thread1, NULL, run_blocking_server, portno);
    pthread_create( &thread2, NULL, run_blocking_client_manager, (void *)&args);
    // if (fork() == 0)
    // {
    //     run_blocking_server(portno);
    // }
    // else
    // {
    //     run_blocking_client("localhost", portno, message);
    // }
    pthread_join( thread1, NULL);
    pthread_join( thread2, NULL);

    end = clock();
    printf("Time passed: %ld", (end - start));
    return 0;
}
