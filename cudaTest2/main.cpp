////////////////////////////////////////////////////////////////////////////
//                                                                        //
// NOTICE TO USER:                                                        //
//                                                                        //
// This file is  part of the CUDA-PSO project which  provides a very fast //
// implementation of  Particle Swarm Optimization within  the nVIDIA CUDA //
// architecture.                                                          //
//                                                                        //
//                                                                        //
// Copyright (C) 2010 IbisLab, University of Parma (Italy)                //
// Authors: Luca Mussi, Fabio Daolio, Stefano Cagnoni                     //
//                                                                        //
//                                                                        //
// cudaPSO  is  free software:  you  can  redistribute  it  and/or modify //
// it under the  terms of the GNU General Public  License as published by //
// the Free Software Foundation, either  version 3 of the License, or (at //
// your option) any later version.                                        //
//                                                                        //
// cudaPSO  is distributed  in  the  hope that  it  will  be useful,  but //
// WITHOUT   ANY  WARRANTY;   without  even   the  implied   warranty  of //
// MERCHANTABILITY  or FITNESS  FOR A  PARTICULAR PURPOSE.   See  the GNU //
// General Public License for more details.                               //
//                                                                        //
// You  should have received  a copy  of the  GNU General  Public License //
// along with cudaPSO. If not, see <http://www.gnu.org/licenses/>.        //
//                                                                        //
//                                                                        //
// If  you do  not agree  to these  terms, do  not download  or  use this //
// software.   For  any  inquiries  different from  license  information, //
// please contact                                                         //
//                                                                        //
// cagnoni@ce.unipr.it and mussi@ce.unipr.it                              //
//                                                                        //
// or                                                                     //
//                                                                        //
// Prof. Stefano Cagnoni / Luca Mussi                                     //
// IbisLab, Dipartimento di Ingegneria dell'Informazione                  //
// Università degli Studi di Parma                                        //
// 181/a Viale G.P. Usberti                                               //
// I-43124 Parma                                                          //
// Italy                                                                  //
//                                                                        //
////////////////////////////////////////////////////////////////////////////


/*!
\mainpage


\latexonly
{
	\begin{center}
	\includegraphics[width=6cm]{IbisLab-logo_small.png} \\
	Intelligent Bio-Inspired Systems Laboratory
	\end{center}
	}
	\vspace{0.7cm}
	\endlatexonly
	
	
	<b>CUDA-PSO Project</b> \n
	\version 1.0
	\author
	<b>IBISlab - Universita' degli Studi di Parma (Italy)</b> \n
	<a href="http://www.ce.unipr.it/people/mussi">Luca Mussi</a> <mussi (at) ce.unipr.it> \n
	<a href="http://www.hec.unil.ch/people/fdaolio">Fabio Daolio</a> <fabio.daolio (at) unil.ch>\n
	<a href="http://www.ce.unipr.it/people/cagnoni">Stefano Cagnoni</a> <cagnoni (at) ce.unipr.it>\n
	
	\image html IbisLab-logo_small.png "Intelligent Bio-Inspired Systems Laboratory"
	
	
	
	
	\section overview Project Aims
	
	Particle Swarm Optimization (PSO) is an iterative process, which, depending on the problem’s complexity,
	may require several thousands (when not millions) of particle updates and fitness evaluations. Therefore,
	designing efficient PSO implementations is an issue of great practical relevance. It becomes even more
	critical if one considers real-time applications to dynamic environments in which, for example, the
	fast-convergence properties of PSO are used to track moving points of interest (maxima or minima of a
	specific dynamically-changing fitness function) in real time. This is the case, for example, of computer
	vision applications in which PSO has been used to track moving objects, or to determine location and
	orientation of objects or posture of people [2,3,4]. This project aims at providing the fastest possible
	implementation of PSO within the
	<a href="http://www.nvidia.com/object/cuda_home_new.html">CUDA</a> architecture by nVidia.
	Futher details about its implementation and tests can be found in [1].
	
	
	
	
	\n \section download Downloads
	
	The sources of this project are available at
	<a href="ftp://ftp.ce.unipr.it/pub/cagnoni/CUDA-PSO/">ftp://ftp.ce.unipr.it/pub/cagnoni/CUDA-PSO/</a>.
	
	
	
	\n \section build_instructions Building the project
	
	The project is shipped as a .tar.gz (or .zip) archive containing the source files.
	These are intended to be built within the nVIDIA CUDA SDK (which should be already
	installed on your system). At the time of writing, the latest CUDA release is the number
	3.2 and all the related downloads can be found at <a href="http://developer.nvidia.com/object/cuda_3_2_toolkit_rc.html" target="_blank">http://developer.nvidia.com/object/cuda_3_2_toolkit_rc.html</a>.
	
	To build the project extract the \c CUDA-PSO folder from the archive and put it into the \c C folder of your CUDA-SDK. Note that the CUDA-PSO folder should be at the same level of the \c src folder containing all the SDK examples for the C language in order for the CUDA compiling mechanism to work correctly.
	
	To build the project use a console and run \c make from the \c RingTopology-3Kernels folder. If everything goes fine, you will find the executable
	\c cudaPSO-Ring-3k in \c CUDA_SDK_FOLDER/C/bin/linux/release.
	
	NOTE 1: Even if the building process went fine, you might get the following error when running the binary executable:
	\verbatim cutilCheckMsg() CUTIL CUDA error: h_init_cudaPSO: cudaBindTexture() execution failed
	in file <cudaPSO.cu>, line 147 : invalid texture reference.\endverbatim
	If this is the case, open the \c Makefile at line 44 and change the flag accordingly to your Streaming Multiprocessors (SMs) version.
	Use
	\verbatim SMVERSIONFLAGS := -arch sm_13 \endverbatim
	if your graphics card is equipped with SMs version 1.3, otherwise use
		\verbatim SMVERSIONFLAGS := -arch sm_11 \endverbatim
		if your graphics card is equipped with SMs version 1.1.
			
			NOTE: So far the project has been developed under the Linux operating system. Anyway, since the code is very general and does not make use of any particular library it should compile and run correctly also under other operating systems.
			
			
			
			
			
			
			\n \section usage_instructions Usage Instructions
			
			The \c cudaPSO-Ring-3k binary is executed from the \c CUDA_SDK_FOLDER/C/bin/linux/release with the following command:
			\verbatim ./cudaPSO-Ring-3k [options]\endverbatim
			
			Alternatively you can run the \c execute bash script available directly into the \c RingTopology-3Kernels folder almost the same way:
			\verbatim ./execute [options]\endverbatim
			
			\subsection usage_options Options
			
			The \c cudaPSO-Ring-3k binary accepts a number of command line switches to
			specify the behaviour of the Particle Swarm and select the problem to be optimized. Specifying no command-line parameters will make the program use default values as detailed below. The switches can be given
			in any order.
			
			\li \c -pf \c \<params_file\> \n \n
			specifies the path of the file to be used to load all the parameters \n \n \n
			
			\li \c -p \c \<particles\> \n \n
			specifies the number of particles (or swarm size)\n
			\e range: positive integer number in [8, 256] \n
			
			\li \c -d \c \<dimension\> \n \n
			specifies the problem dimensionality\n
			\e range: positive integer number in [2, 512] \n \n \n
			
			\li \c -g \c \<generations\> \n \n
			specifies the number of generations\n
			\e range: positive integer number greater than 0 (at least 1) \n \n \n
			
			\li \c -r \c \<runs\> \n \n
			specifies the number of times for the optimization to be repeated\n
			\e range: positive integer number greater than 0 (at least 1) \n \n \n
			
			\li \c -f \c \<function\> \n \n
			specifies the fitness function\n
			\e range: positive integer number greater than 0 (at the moment only three different functions are implemented, so valid values are 0, 1 and 2 only) \n \n \n
			
			
			\section requirements Requirements
			The program requires an NVIDIA CUDA-compatible graphic card (GeForce7 series or better) as well as the CUDA toolkit installed on your machine.
			
			
			\n \section references References
			
			\htmlonly
			
			1. Evaluation of Parallel Particle Swarm Optimization Algorithms within the CUDA Architecture<br>
			Luca Mussi, Fabio Daolio, Stefano Cagnoni<br>
			Information Science (2010), in press<br>
			DOI:  <a href="http://dx.doi.org/10.1016/j.ins.2010.08.045">10.1016/j.ins.2010.08.045</a><br>
			<br>
			
			2. Markerless Articulated Human Body Tracking from Multi-View Video with GPU-PSO<br>
			Luca Mussi, Spela Ivekovic, Stefano Cagnoni<br>
			9th International Conference on Evolvable Systems (<a href="http://www.ices2010.org/">ICES 2010</a>)<br>
			ISBN:  <a href="http://www.springer.com/computer/swe/book/978-3-642-15322-8?changeHeader">978-3-642-15322-8</a><br>
			Download: <a href="http://www.ce.unipr.it/people/mussi/downloads/papers/mussiICES10.pdf">Pdf</a> (340 Kb) - <a href="http://www.ce.unipr.it/people/mussi/downloads/papers/mussiICES10.bib">BibTeX</a><br>
			<br>
			
			
			3. GPU Implementation of a Road Sign Detector Based on Particle Swarm Optimization<br>
			Luca Mussi and Stefano Cagnoni and Elena Cardarelli and Fabio Daolio and Paolo Medici and Pier Paolo Porta<br>
			Evolutionary Intelligence (2010), Volume 3, Numbers 3-4, 155-169<br>
			DOI:  <a href="http://dx.doi.org/10.1007/s12065-010-0043-y">10.1007/s12065-010-0043-y</a><br>
			<br>
			
			
			4.GPU-based Road Sign Detection Using Particle Swarm Optimization<br>
			Luca Mussi, Fabio Daolio, Stefano Cagnoni<br>
			9th International Conference on Intelligent Systems Design and Applications (<a href="http://cig.iet.unipi.it/isda09/">ISDA 2009</a>)<br>
			DOI: <a href="http://dx.doi.org/10.1109/ISDA.2009.88">10.1109/ISDA.2009.88</a><br>
			Download: <a href="http://www.ce.unipr.it/people/mussi/downloads/papers/mussiISDA09.pdf">Pdf</a> (329 Kb) - <a href="http://www.ce.unipr.it/people/mussi/downloads/papers/mussiISDA09.bib">BibTeX</a><br>
			\endhtmlonly
			
			
			
			\latexonly
			
			\hangindent=0.4cm
			1. Mussi, L., Daolio, F., Cagnoni, S.:
			\textit{Evaluation of parallel particle swarm optimization algorithms within the cuda architecture.}
			Information Sciences (2010). DOI 10.1016/j.ins.2010.08.045
			
			
			\hangindent=0.4cm
			2. Mussi, L., Ivekovic, S., Cagnoni, S.:
			\textit{Markerless articulated human body tracking from multi-view video with GPU-PSO.}
			In: G. Tempesti, A.M. Tyrrell, J.F. Miller (eds.) Evolvable Systems: From Biology
			to Hardware - 9th International Conference, ICES 2010 York, UK, September 2010 Proceedings,
			no. 6274 in Lecture Notes in Computer Science, pp. 195–207. Springer, Berlin, Heidelberg (2010)
			
			
			\hangindent=0.4cm
			3. Mussi, L., Cagnoni, S., Cardarelli, E., Daolio, F., Medici, P., Porta, P.P.:
			\textit{GPU implementation of a road sign detector based on particle swarm optimization.}
			Evolutionary Intelligence (2010), Volume 3, Numbers 3-4, 155-169. DOI 10.1007/s12065-010-0043-y
			
			
			\hangindent=0.4cm
			4. Mussi, L., Daolio, F., Cagnoni, S.:
			\textit{GPU-based road sign detection using particle swarm optimization.}
			In: Proc. IEEE Conference on Intelligent System Design and Applications (ISDA09), pp. 152–157.
			IEEE CS Press, Washington, DC, USA (2009)
			
			\endlatexonly
			
			*/


#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
//#include <cutil_inline.h>

//#include <helper_cuda.h>
#include <cstdio>
//#include <helper_string.h>


#include "cudaPSO.h"
#include "cutil_compat.h"

using namespace std;

///default parameters file (commandline parameter -pf overrides it)
#define PARAMETERS_FILE "data/PSO-parameters-Sphere.txt"

//NOTE To print debug messages uncomment the following line
//#define PRINT_DEBUG


void parseCommandlineParameters(int argc, char** argv, int* swarmsNumber, int* particlesNumber, int* problemDimension, int* numberOfGenerations, int* functionID, int* numberOfRuns);
void getParametersFile(int argc, char** argv, string &parametersFile);


int main(int argc, char **argv)
{
	//Initialization of CUDA
	cudaSetDevice( cutGetMaxGflopsDeviceId() );


	//This version is for one swarm only...
	int swarmsNumber = -1;

	int particlesNumber = -1;
	int problemDimension = -1;
	int numberOfGenerations = -1;
	int functionID = -1;
	int numberOfRuns = 1;
	string parametersFile = PARAMETERS_FILE;

	getParametersFile(argc, argv, parametersFile);
	parseCommandlineParameters(argc, argv, &swarmsNumber, &particlesNumber, &problemDimension, &numberOfGenerations, &functionID, &numberOfRuns);

	cudaPSO swarm( parametersFile );

	if (particlesNumber     != -1) swarm.setNumberOfParticles(particlesNumber);
	if (problemDimension    != -1) swarm.setProblemDimension(problemDimension);
	if (numberOfGenerations != -1) swarm.setNumberOfGenerations(numberOfGenerations);
	if (functionID          != -1) swarm.setFunctionID(functionID);

	swarm.init();

	for(int i = 0; i < numberOfRuns; ++i){
		swarm.optimize();
		swarm.printResults();
	}

	cout << "Done!" << endl;


	return 0;
}


/// @brief Parses the commadline looking for the parameters file name
/// @param parametersFile string to wich return the name of the parameter file
void getParametersFile(int argc, char** argv, string &parametersFile){
	char *stringValue;

	//if( cutCheckCmdLineFlag(argc, (const char**)argv, "pf") ){
	//	#ifdef PRINT_DEBUG
	//		printf("checking parameter pf...\n");
	//	#endif
	//	cutGetCmdLineArgumentstr( argc, (const char**) argv, "pf", &stringValue);
	//	parametersFile = stringValue;
	//}
	//#ifdef PRINT_DEBUG
	//	printf("parametersFile = %s!\n", parametersFile.c_str());
	//#endif
}


/// @brief Parses commadline parameters
/// @param swarmsNumber pointer to wich return the number of swarms
/// @param particlesNumber pointer to wich return the number of particles per swarm
/// @param problemDimension pointer to wich return the dimension of the search space
/// @param numberOfGenerations pointer to wich return the number of generations to be performed during optimization
/// @param functionID pointer to wich return the ID of the function to be optimized
/// @param numberOfRuns pointer to wich return the number of trials/runs to be performed
void parseCommandlineParameters(int argc, char** argv, int* swarmsNumber, int* particlesNumber, int* problemDimension, int* numberOfGenerations, int* functionID, int* numberOfRuns){
//	int   intValue;
//
//	//This version is for only one swarm...
//	//if( cutCheckCmdLineFlag(argc, (const char**)argv, "s") ){
//	//	#ifdef PRINT_DEBUG
//	//		printf("checking parameter s...\n");
//	//	#endif
//	//	if (	cutGetCmdLineArgumenti( argc, (const char**) argv, "s", &intValue) && (intValue > 0) )
//	// 		*swarmsNumber = intValue;
//	// 	else
//	// 		printf("the number of swarms must be a valid positive number! assuming %d...\n", *swarmsNumber);
//	// 	}
//
//	//This version is for only one swarm...
//	*swarmsNumber = 1;
//	
//	#ifdef PRINT_DEBUG
//		printf("s = %d!\n", *swarmsNumber);
//	#endif
//
//	if( cutCheckCmdLineFlag(argc, (const char**)argv, "p") ){
//		#ifdef PRINT_DEBUG
//			printf("checking parameter p...\n");
//		#endif
//		if (	cutGetCmdLineArgumenti( argc, (const char**) argv, "p", &intValue) &&
//			(intValue >= 8) && (intValue <=256)
//		   )
//			*particlesNumber = intValue;
//		else{
//			printf("the number of particles must be a valid positive number in [8, 256]!");
//		}
//	}
//	#ifdef PRINT_DEBUG
//		printf("p = %d!\n", *particlesNumber);
//	#endif
//
//	if( cutCheckCmdLineFlag(argc, (const char**)argv, "d") ){
//		#ifdef PRINT_DEBUG
//			printf("checking parameter d...\n");
//		#endif
//		if(	cutGetCmdLineArgumenti( argc, (const char**) argv, "d", &intValue) &&
//			(intValue > 1) && (intValue <= 512)
//		  )
//			*problemDimension = intValue;
//		else{
//			printf("the number of problem's dimension must be a valid positive number in [2, 512]!\n");
//			exit(0);
//		}
//	}
//	#ifdef PRINT_DEBUG
//		printf("d = %d!\n", *problemDimension);
//	#endif
//
//	if( cutCheckCmdLineFlag(argc, (const char**)argv, "g") ){
//		#ifdef PRINT_DEBUG
//			printf("checking parameter g...\n");
//		#endif
//		if (	cutGetCmdLineArgumenti( argc, (const char**) argv, "g", &intValue) &&
//			(intValue > 0)
//		   )
//			*numberOfGenerations = intValue;
//		else
//			printf("the number of generations must be a valid positive number! assuming %d...\n", *numberOfGenerations);
//	}
//	#ifdef PRINT_DEBUG
//		printf("g = %d!\n", *numberOfGenerations);
//	#endif
//
//	if( cutCheckCmdLineFlag(argc, (const char**)argv, "f") ){
//		#ifdef PRINT_DEBUG
//			printf("checking parameter f...\n");
//		#endif
//		if (	cutGetCmdLineArgumenti( argc, (const char**) argv, "f", &intValue) &&
//			(intValue >= 0)
//		   )
//			*functionID = intValue;
//		else
//			printf("the index of the function to be optimized must be a valid positive number! assuming %d...\n", *functionID);
//	}
//	#ifdef PRINT_DEBUG
//		printf("r = %d!\n", *functionID);
//	#endif
//
//	if( cutCheckCmdLineFlag(argc, (const char**)argv, "r") ){
//		#ifdef PRINT_DEBUG
//			printf("checking parameter r...\n");
//		#endif
//		if (	cutGetCmdLineArgumenti( argc, (const char**) argv, "r", &intValue) &&
//			(intValue >= 0)
//		   )
//			*numberOfRuns = intValue;
//		else
//			printf("the number of runs must be a valid positive number! assuming %d...\n", *numberOfRuns);
//	}
//	#ifdef PRINT_DEBUG
//		printf("r = %d!\n", *numberOfRuns);
//	#endif
}



