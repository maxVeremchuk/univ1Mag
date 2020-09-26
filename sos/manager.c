#include <time.h>

#include "blocking_client.c"
#include "blocking_server.c"

int main(int argc, char* argv[])
{
    const int portno = 51717;
    char* message = "message message";
    time_t start, end;
    start = clock();

    if (fork() == 0)
    {
        run_blocking_server(portno);
    }
    else
    {
        run_blocking_client("localhost", portno, message);
    }

    end = clock();
    printf("Time passed: %ld", (end - start));
    return 0;
}
