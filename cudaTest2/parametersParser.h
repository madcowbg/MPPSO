
#ifndef _PARAMETERS_PARSER_H
#define _PARAMETERS_PARSER_H


/**
  @brief Class for parsing a simple plain text file.

 Used to load the various parameters of the algorithm such as
 the number of the particle in the Swarm, the inertia weight,
 the function to be optimized, etc.

 This class is actually written in old C-style and is not really optimized for speed.

\section param_file_example Example of a parameter file:

\verbatim
//This is a comment
//A Parameter starts with a '#' at the very beginning of the line
//the name of the parameter must follows the '#' without any space!

#BASE_DIR /home/user/   //this is another comment...
#COEFF 3.2
#IMG_WIDTH 720
\endverbatim

\section parsing_example Example of a function to parse parameters:

\verbatim
bool parseAllParameters(char* fileName, std::string& baseDir, double& coeff, int& width){
	
	double doubleValue = 0;
	int intValue = 0;
	char tmp[100];
	
	
	if(simpleParse(fileName, "BASE_DIR", tmp))
		baseDir = tmp;
	else
		return false;
	
	if(simpleParse(fileName, "COEFF", &doubleValue))
		coeff = doubleValue;
	else
		return false;
	
	if(simpleParse(fileName, "IMG_WIDTH", &intValue))
		width = intValue;
	else
		return false;
	
	return true;
}
\endverbatim
*/

class parametersParser{
  public:

	//! @brief Parse one single parameter of type \c double
	//! @param fileName path to the text file to be parsed
	//! @param parameterName name of the parameter to be loaded from file (case sensitive)
	//! @param par pointer to the variable to contain the loaded value
	bool parse(const char* fileName, const char* parameterName, double* par);

	//! @brief Parse one single parameter of type \c int
	//! @param fileName path to the text file to be parsed
	//! @param parameterName name of the parameter to be loaded from file (case sensitive)
	//! @param par pointer to the variable to contain the loaded value
	bool parse(const char* fileName, const char* parameterName, int*    par);

	//! @brief Parse one single parameter of type \c string
	//! @param fileName path to the text file to be parsed
	//! @param parameterName name of the parameter to be loaded from file (case sensitive)
	//! @param par pointer to the variable to contain the loaded value
	bool parse(const char* fileName, const char* parameterName, char*   par);
};





#endif


