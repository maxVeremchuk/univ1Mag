#include "datautils.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define BUF_SIZE 8192
#define SHM_KEY 0x1234

struct shmseg
{
    int cnt;
    int complete;
    char buf[BUF_SIZE];
};

void write_shared_memory_data()
{
    int shmid, numtimes;
    struct shmseg* shmp;
    char* bufptr;
    int spaceavailable;
    shmid = shmget(SHM_KEY, sizeof(struct shmseg), 0644 | IPC_CREAT);
    if (shmid == -1)
    {
        perror("Shared memory");
        exit(1);
    }

    shmp = shmat(shmid, NULL, 0);
    if (shmp == (void*)-1)
    {
        perror("Shared memory attach");
        exit(1);
    }

    bufptr = shmp->buf;
    spaceavailable = BUF_SIZE;
    for (numtimes = 0; numtimes < 100000; numtimes++)
    {
        strcpy(bufptr, c_message);
        shmp->cnt = strlen(bufptr);
        shmp->complete = 0;
        bufptr = shmp->buf;
        spaceavailable = BUF_SIZE;
    }
    shmp->complete = 1;

    if (shmdt(shmp) == -1)
    {
        perror("shmdt");
        exit(1);
    }

    if (shmctl(shmid, IPC_RMID, 0) == -1)
    {
        perror("shmctl");
        exit(1);
    }
}

void read_shared_memory_data()
{
    int shmid;
    struct shmseg* shmp;
    shmid = shmget(SHM_KEY, sizeof(struct shmseg), 0644 | IPC_CREAT);
    if (shmid == -1)
    {
        perror("Shared memory");
        exit(1);
    }

    shmp = shmat(shmid, NULL, 0);
    if (shmp == (void*)-1)
    {
        perror("Shared memory attach");
        exit(1);
    }

    while (shmp->complete != 1)
    {
        //printf("segment contains : \n\"%s\"\n", shmp->buf);
        if (shmp->cnt == -1)
        {
            perror("read");
            exit(1);
        }

    }

    if (shmdt(shmp) == -1)
    {
        perror("shmdt");
        exit(1);
    }
}

int main(int argc, char* argv[])
{
    pthread_t thread1, thread2;
    time_t start, end;

    start = clock();

    pthread_create(&thread1, NULL, write_shared_memory_data, NULL);
    pthread_create(&thread2, NULL, read_shared_memory_data, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    end = clock();

    printf("Time passed shared mem: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    return 0;
}
