#include <iostream>
#include <string>
#include "PSO.cuh"
#include "IndStudentTPSO.cuh"
#include "ParallelPSO.h"
#include "FindIndTZeroDeriv.cuh"

#include <ctime>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

void readFile(ParallelPSO& ppso, const char* filename);

int main(int argc, char* argv[]) {
	ParallelPSO ppso;

	if (argc == 3 && strcmp(argv[1], "-file") == 0) {
		readFile(ppso, argv[2]);
	} else {
		//cout<<argc << "other params" << endl;
		gpu_fp_t inertia = 0.3;
		gpu_fp_t cpbest = 0.3;
		gpu_fp_t cgbest = 0.3;
		gpu_fp_t vmax = 1;

		gpu_fp_t nu = 0;
		int nParticles = 8;
		int nIter = 200;
	
		gpu_fp_t sampleElem;
		vector<gpu_fp_t> sample;

		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-data") == 0) {
				//std::cout << "reading from " << argv[i] << std::endl;
				std::stringstream ss(argv[++i]);
				while (!ss.eof()) {
					ss >> sampleElem;
					sample.push_back(sampleElem);
				}
			} else if (strcmp(argv[i], "-nu") == 0) {
				std::stringstream ss(argv[++i]);
				ss >> nu;
			} else if (strcmp(argv[i], "-inertia") == 0) {
				std::stringstream ss(argv[++i]);
				ss >> inertia;
			} else if (strcmp(argv[i], "-cpbest") == 0) {
				std::stringstream ss(argv[++i]);
				ss >> cpbest;
			} else if (strcmp(argv[i], "-cgbest") == 0) {
				std::stringstream ss(argv[++i]);
				ss >> cgbest;
			} else if (strcmp(argv[i], "-vmax") == 0) {
				std::stringstream ss(argv[++i]);
				ss >> vmax;
			} else if (strcmp(argv[i], "-nIter") == 0) {
				std::stringstream ss(argv[++i]);
				ss >> nIter;
			} else {
				cerr << "Wrong param: " << argv[i] << endl;
				return 1;
			}
		}
		//for (unsigned int i = 0; i < sample.size(); i++)
		//	cerr<<sample[i]<<" ";
		//cerr<<endl<<nu<<" "<<endl;

		IndStudentTPSO* fitter = new IndStudentTPSO(sample.data(), sample.size(), nu, nParticles, nIter);
		fitter->setParams(psoopts(inertia, cpbest, cgbest, vmax));
		ppso.add(fitter);
	}

	// run optimization
	ppso.optimize();

	// write results
	for (unsigned int i = 0; i < ppso.size(); i++) {
		IndStudentTPSO* fitter = (IndStudentTPSO*)ppso.get(i);
		for (int iTask = 0; iTask < fitter->getNTasks(); iTask ++ ){
			// output
			cout << fitter->getFit(iTask) << endl;
			cout << fitter->getMu(iTask) << endl;
			cout << fitter->getSigma(iTask) << endl;
		}
		/*FindIndTZeroDeriv* fitter = (FindIndTZeroDeriv*)ppso.get(i);
		for (int iTask = 0; iTask < fitter->getNTasks(); iTask ++ ){
			cout << fitter->getFit(iTask) << endl;
			cout << fitter->getResult(iTask, 0) << endl;
			cout << fitter->getResult(iTask, 1) << endl;
		}*/
	}
	//	fitter.optimize();

	return 0;
}

void readFile(ParallelPSO& ppso, const char* filename) {
	gpu_fp_t nu = 0;
	int nParticles = 5;
	int nIter = 200;
	psoopts opts(1,2,2,2);

	vector<vector<gpu_fp_t> > samples;
	ifstream fs(filename);
	while (!fs.eof()) {
		string params;
		getline(fs, params);
		stringstream sparams(params);


		while (!sparams.eof()) {
			string param;
			sparams >> param;
			
			if (sparams.eof()) {
				break;
			}

			if (param == "-nu") {
				sparams >> nu;
			} else if (param == "-inertia") {
				sparams >> opts.inertia;
			} else if (param == "-cpbest") {
				sparams >> opts.cpbest;
			} else if (param == "-cgbest") {
				sparams >> opts.cgbest;
			} else if (param == "-vmax") {
				sparams >> opts.vmax;
			} else if (param == "-nIter") {
				sparams >> nIter;
			} else if (param == "-nPart") {
				sparams >> nParticles;
			} else {
				cerr << "Wrong param: " << param << endl;
			}
		}
//		nParticles=4;
//		opts.cpbest = 1.4;
//		opts.cgbest = 1.4;
//		opts.inertia = 0.1;

		if (fs.eof()) {
			break;
		}

		string data;
		getline(fs, data);
		stringstream sdata(data);
		vector<gpu_fp_t> sample;
		while (!sdata.eof()) {
			gpu_fp_t sampleElem;
			sdata >> sampleElem;
			sample.push_back(sampleElem);
		}

		samples.push_back(sample);
	}

	int sampleSize = 0;
	vector<gpu_fp_t*> sampleVectors;
	for (int i = 0; i < samples.size(); i++) {//samples.size()
		sampleVectors.push_back(samples[i].data());
		sampleSize = samples[i].size();
	}

	IndStudentTPSO* fitter = new IndStudentTPSO(sampleVectors, sampleSize, nu, nParticles, nIter);
	fitter->setParams(opts);
	//cout << sampleSize << endl;
	//FindIndTZeroDeriv* fitter = new FindIndTZeroDeriv(sampleVectors, sampleSize, nu, 0, 1, nParticles, nIter);
	//fitter->setParams(opts);

	ppso.add(fitter);
}