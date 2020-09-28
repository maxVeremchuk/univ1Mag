#include <stdio.h>
#include <string.h>

#include "aes.h"

void readInput(char []);

// this function reads the user input
void readInput(char input[BYTES+1]){
	unsigned short int sizeCheck;	// temporary saving the length of given input
	
	do{
		// read the input
		gets(input);
		
		// clear the input buffer
		fflush(stdin);
		
		// save the length of input
		sizeCheck = strlen(input);
	}while(sizeCheck != BYTES);
	
return;
}
