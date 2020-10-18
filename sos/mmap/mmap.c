#include "datautils.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

void write_mmap_sample_data()
{
    int fd;
    char ch;
    struct stat textfilestat;
    fd = open("MMAP_DATA.txt", O_CREAT | O_TRUNC | O_WRONLY, 0666);
    if (fd == -1)
    {
        perror("File open error ");
        exit(1);
    }

    write(fd, c_message, strlen(c_message));

    close(fd);
}

void read_mmap_sample_data()
{
    struct stat mmapstat;
    char* data;
    int minbyteindex;
    int maxbyteindex;
    int offset;
    int fd;
    int unmapstatus;

    if (stat("MMAP_DATA.txt", &mmapstat) == -1)
    {
        perror("stat failure");
        exit(1);
    }

    if ((fd = open("MMAP_DATA.txt", O_RDONLY)) == -1)
    {
        perror("open failure");
        exit(1);
    }
    data = mmap((caddr_t)0, mmapstat.st_size, PROT_READ, MAP_SHARED, fd, 0);

    if (data == (caddr_t)(-1))
    {
        perror("mmap failure");
        exit(1);
    }
    minbyteindex = 0;
    maxbyteindex = mmapstat.st_size - 1;

    printf("%s\n", data);

    unmapstatus = munmap(data, mmapstat.st_size);

    if (unmapstatus == -1)
    {
        perror("munmap failure");
        exit(1);
    }
    close(fd);
    system("rm -f MMAP_DATA.txt");
}

int main()
{
    time_t start, end;

    start = clock();
    for (int i = 0; i < 1000; i++)
    {
        write_mmap_sample_data();
        read_mmap_sample_data();
    }
    end = clock();

    printf("Time passed mmap: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    return 0;
}
