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
        return 1;
    }

    // Attach to the segment to get a pointer to it.
    shmp = shmat(shmid, NULL, 0);
    if (shmp == (void*)-1)
    {
        perror("Shared memory attach");
        return 1;
    }

    /* Transfer blocks of data from buffer to shared memory */
    bufptr = shmp->buf;
    spaceavailable = BUF_SIZE;
    for (numtimes = 0; numtimes < 5; numtimes++)
    {
        strcpy(bufptr, c_message);
        shmp->cnt = strlen(bufptr);
        shmp->complete = 0;
        printf("Writing Process: Shared Memory Write: Wrote %d bytes\n", shmp->cnt);
        bufptr = shmp->buf;
        spaceavailable = BUF_SIZE;
        sleep(3);
    }
    printf("Writing Process: Wrote %d times\n", numtimes);
    shmp->complete = 1;

    if (shmdt(shmp) == -1)
    {
        perror("shmdt");
        return 1;
    }

    if (shmctl(shmid, IPC_RMID, 0) == -1)
    {
        perror("shmctl");
        return 1;
    }
    printf("Writing Process: Complete\n");
}

void read_shared_memory_data()
{
    int shmid;
    struct shmseg* shmp;
    shmid = shmget(SHM_KEY, sizeof(struct shmseg), 0644 | IPC_CREAT);
    if (shmid == -1)
    {
        perror("Shared memory");
        return 1;
    }

    // Attach to the segment to get a pointer to it.
    shmp = shmat(shmid, NULL, 0);
    if (shmp == (void*)-1)
    {
        perror("Shared memory attach");
        return 1;
    }

    /* Transfer blocks of data from shared memory to stdout*/
    while (shmp->complete != 1)
    {
        printf("segment contains : \n\"%s\"\n", shmp->buf);
        if (shmp->cnt == -1)
        {
            perror("read");
            return 1;
        }
        printf("Reading Process: Shared Memory: Read %d bytes\n", shmp->cnt);
        sleep(3);
    }
    printf("Reading Process: Reading Done, Detaching Shared Memory\n");
    if (shmdt(shmp) == -1)
    {
        perror("shmdt");
        return 1;
    }
    printf("Reading Process: Complete\n");
}

int main(int argc, char* argv[])
{
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, write_shared_memory_data, NULL);
    pthread_create(&thread2, NULL, read_shared_memory_data, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    return 0;
}
