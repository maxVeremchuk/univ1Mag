#include <stdio.h>

#include "aes.h"

int main(){
	// last extra byte is for '\0' character
	// unsigned char plainText[BYTES+1];	// plaintext
	// unsigned char cipherText[BYTES+1];	// ciphertext
	// unsigned char key[BYTES+1];			// encryption key
	// unsigned char output[BYTES+1];		// string to prints the results

	// int answer;		// user answer

	// unsigned int state4x4[BLOCK_SIZE][BLOCK_SIZE];	// input
	// unsigned int key4x4[BLOCK_SIZE][BLOCK_SIZE];	// encryption or decryption key

	// we have 10 round, so we need 40 words in array plus 4 for the given key
	// each word have 4 bytes, so we need 44 * 4 = 176
	unsigned int w[176];		// all round keys
	unsigned int state[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x4f, 0x4e, 0x20},
                                                  {0x77, 0x6e, 0x69, 0x54},
                                                  {0x6f, 0x65, 0x6e, 0x77},
                                                  {0x20, 0x20, 0x65, 0x6f}};
	unsigned int key[BLOCK_SIZE][BLOCK_SIZE] = {{0x54, 0x73, 0x20, 0x67},
                                                {0x68, 0x20, 0x4b, 0x20},
                                                {0x61, 0x6d, 0x75, 0x46},
                                                {0x74, 0x79, 0x6e, 0x75}};
	unsigned int roundKey[BLOCK_SIZE][BLOCK_SIZE];
	key_schedule(w, key);

	getRoundKey(w, roundKey, 0);
    add_roundkey(state, state, roundKey);
	for (int i = 0; i < BLOCK_SIZE; ++i)
    {
		for (int j = 0; j < BLOCK_SIZE; ++j)
		{
        	printf("%d, ", roundKey[i]);
		}
    }

	// print the menu to user and read the answer
	// do{
	// 	printf("Choose action: ");
	// 	scanf("%d", &answer);

	// }while(answer < 1 || answer > 4);

	// switch(answer){
	// 	case 1:
	// 		// read the plaintext
	// 		printf("Text: ");
	// 		readInput(plainText);

	// 		// read the key
	// 		printf("Key: ");
	// 		readInput(key);

	// 		// convert plaintext and key to 4x4 blocks
	// 		convertStringToBlock(plainText, state4x4);
	// 		convertStringToBlock(key, key4x4);

	// 		// we calculate all round keys 1-10
	// 		key_schedule(w, key4x4);

	// 		// encrypt the plaintext
	// 		encryption(state4x4, w);

	// 		// print the output to user as string
	// 		convertBlockToString(state4x4, output);
	// 		output[BYTES] = '\0';
	// 		printf("Ciphertext: %s", output);
	// 		break;

	// 	case 2:
	// 		// read the ciphertext
	// 		printf("Text: ");
	// 		readInput(cipherText);

	// 		// read the key
	// 		printf("Key: ");
	// 		readInput(key);

	// 		// convert ciphertext and key to 4x4 blocks
	// 		convertStringToBlock(cipherText, state4x4);
	// 		convertStringToBlock(key, key4x4);

	// 		// we calculate all round keys 1-10
	// 		key_schedule(w, key4x4);

	// 		// decrypt the ciphertext
	// 		decryption(state4x4, w);

	// 		// print the output to user as string
	// 		convertBlockToString(state4x4, output);
	// 		output[BYTES] = '\0';
	// 		printf("Plaintext: %s", output);
	// 		break;

	// 	case 3:
	// 		// read the plaintext
	// 		printf("Text: ");
	// 		readInput(plainText);

	// 		// read the key
	// 		printf("Key: ");
	// 		readInput(key);

	// 		// convert plaintext and key to 4x4 blocks
	// 		convertStringToBlock(plainText, state4x4);
	// 		convertStringToBlock(key, key4x4);

	// 		// we calculate all round keys 1-10
	// 		key_schedule(w, key4x4);

	// 		printf("\nENCRYPTION\n");
	// 		// encrypt the plaintext
	// 		encryption(state4x4, w);

	// 		// print the output to user as string
	// 		convertBlockToString(state4x4, output);
	// 		output[BYTES] = '\0';
	// 		printf("Ciphertext: %s\n", output);
	// 		printf("END OF ENCRYPTION\n");

	// 		printf("\n\nDECRYPTION\n");
	// 		// decrypt the ciphertext
	// 		decryption(state4x4, w);

	// 		// print the output to user as string
	// 		convertBlockToString(state4x4, output);
	// 		output[BYTES] = '\0';
	// 		printf("Plaintext: %s\n", output);
	// 		printf("END OF DECRYPTION\n");
	// 		break;

	// 	case 4:
	// 		test();
	// 		break;
	// }

return 0;
}
