
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parametersParser.h"


bool parametersParser::parse(const char* fileName, const char* parameterName, double* par){

	FILE *fp;
	if ((fp = fopen(fileName, "r")) == NULL){
		printf("\nError opening file %s!\n", fileName);
		return false;
	}

	char buffer[512];
	char* tok = 0;
	char* value = 0;
	while(fgets(buffer,512,fp)){
		tok = strchr(buffer,'#');
		if(tok){
			tok++;
			value = strchr(tok,' ');
			if (value){
				*value++ = 0;
				if(!strcmp(tok, parameterName)){
					if(!sscanf(value, "%lf", par)){
						fclose(fp);
						printf("%s = %s: invalid format for parameter value!!!\n", parameterName, value);
						return false;
					}
					fclose(fp);
					return true;
				}
			}
		}
	}
	fclose(fp);
	printf("Parameter %s not found!!!\n", parameterName);
	return false;
}


bool parametersParser::parse(const char* fileName, const char* parameterName, int* par){

	FILE *fp;
	if ((fp = fopen(fileName, "r")) == NULL){
		printf("\nError opening file %s!\n", fileName);
		return false;
	}

	char buffer[512];
	char* tok = 0;
	char* value = 0;
	while(fgets(buffer,512,fp)){
		tok = strchr(buffer,'#');
		if(tok){
			tok++;
			value = strchr(tok,' ');
			if (value){
				*value++ = 0;
				if(!strcmp(tok, parameterName)){
					if(!sscanf(value, "%d", par)){
						fclose(fp);
						printf("%s = %s: invalid format for parameter value!!!\n", parameterName, value);
						return false;
					}
					fclose(fp);
					return true;
				}
			}
		}
	}
	fclose(fp);
	printf("Parameter %s not found!!!\n", parameterName);
	return false;
}

bool parametersParser::parse(const char* fileName, const char* parameterName, char* par){

	FILE *fp;
	if ((fp = fopen(fileName, "r")) == NULL){
		printf("\nError opening file %s!\n", fileName);
		return false;
	}

	char buffer[512];
	char* tok = 0;
	char* value = 0;
	while(fgets(buffer,512,fp)){
		tok = strchr(buffer,'#');
		if(tok){
			tok++;
			value = strchr(tok,' ');
			if (value){
				*value++ = 0;
				if(!strcmp(tok, parameterName)){
					if(!sscanf(value, "%s", par)){
						fclose(fp);
						printf("%s = %s: invalid format for parameter value!!!\n", parameterName, value);
						return false;
					}
					fclose(fp);
					return true;
				}
			}
		}
	}
	fclose(fp);
	printf("Parameter %s not found!!!\n", parameterName);
	return false;
}


