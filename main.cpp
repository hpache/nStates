#include <Eigen/Sparse>
#include <iostream>
#include <complex>
#include <fstream>
#include <chrono>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;

// First we have the values we need for the weight (Maxwell-Boltzmann Distribution)
double kB = 1.381e-23;
double AMU = 1.66e-27;
double mAtom = 39;
double temperature = 323;
double cellLength = 7.0; // This is in cm
double u = sqrt(2 * kB * temperature / (mAtom * AMU)) * 100;
double logP = 9.967 - 4646 / temperature;
double pressure = pow(10, logP);
double density = (pressure) / (kB * temperature) * 1e-06;

double delta_p;

double Omega_01;
double Omega_12;
double Omega_13;

double Delta_23;

double k_p;
double k_c;

double Gamma_21;
double Gamma_31;
double Gamma_10;

double gamma_p;
double gamma_c;

double lambdaP;
double lambdaC;
double alphaP_0;
double alphaC_0;

double initialDetuning;
double finalDetuning;
double numberDetunings;

double initialVelocity;
double finalVelocity;
double numberVelocity;

std::string fileName;

SparseLU<SparseMatrix<dcomplex>, COLAMDOrdering<int> > solver;

SparseMatrix<dcomplex> B;
SparseVector<dcomplex> x(15);
SparseVector<dcomplex> b(15);

SparseMatrix<dcomplex> M(double delta_c, double v) {

	SparseMatrix<dcomplex> A(15, 15);
	A.coeffRef(0, 0) = dcomplex(-1.0 * Gamma_10, 0);
	A.coeffRef(0, 1) = dcomplex(Gamma_21, 0);
	A.coeffRef(0, 2) = dcomplex(Gamma_31, 0);
	A.coeffRef(0, 3) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(0, 6) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(0, 7) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(0, 8) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(0, 10) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(0, 13) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(1, 1) = dcomplex(-1.0 * Gamma_21, 0);
	A.coeffRef(1, 7) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(1, 10) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(2, 2) = dcomplex(-1.0 * Gamma_31, 0);
	A.coeffRef(2, 8) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(2, 13) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(3, 0) = dcomplex(0, -1.0 * Omega_01);
	A.coeffRef(3, 1) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(3, 2) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(3, 3) = dcomplex(-0.5 * Gamma_10 + gamma_p, -1.0 * (-delta_p - k_p * v));
	A.coeffRef(3, 4) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(3, 5) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(4, 3) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(4, 4) = dcomplex(-0.5 * Gamma_21 - gamma_c - gamma_p, 1.0 * (delta_c - k_c * v - (-delta_p - k_p * v)));
	A.coeffRef(4, 7) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(5, 3) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(5, 5) = dcomplex(-0.5 * Gamma_31 - gamma_c - gamma_p, 1.0 * (Delta_23 + delta_c - k_c * v - (-delta_p - k_p * v)));
	A.coeffRef(5, 8) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(6, 0) = dcomplex(0, 1.0 * Omega_01);
	A.coeffRef(6, 1) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(6, 2) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(6, 6) = dcomplex(-0.5 * Gamma_10 - gamma_p, 1.0 * (-delta_p - k_p * v));
	A.coeffRef(6, 9) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(6, 12) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(7, 0) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(7, 1) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(7, 4) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(7, 7) = dcomplex(-0.5 * Gamma_10 - 0.5 * Gamma_21 - gamma_c, 1.0 * (delta_c - k_c * v));
	A.coeffRef(7, 14) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(8, 0) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(8, 2) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(8, 5) = dcomplex(0, -0.5 * Omega_01);
	A.coeffRef(8, 8) = dcomplex(-0.5 * Gamma_10 - 0.5 * Gamma_31 - gamma_c, 1.0 * (Delta_23 + delta_c - k_c * v));
	A.coeffRef(8, 11) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(9, 6) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(9, 9) = dcomplex(-0.5 * Gamma_21 - gamma_c - gamma_p, 1.0 * (-delta_p - k_p * v - (delta_c - k_c * v)));
	A.coeffRef(9, 10) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(10, 0) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(10, 1) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(10, 9) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(10, 10) = dcomplex(-0.5 * Gamma_10 - 0.5 * Gamma_21 - gamma_c, -1.0 * (delta_c - k_c * v));
	A.coeffRef(10, 11) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(11, 8) = dcomplex(0, -0.5 * Omega_12);
	A.coeffRef(11, 10) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(11, 11) = dcomplex(-0.5 * Gamma_21 - 0.5 * Gamma_31, 1.0 * (Delta_23 + delta_c - k_c * v - (delta_c - k_c * v)));
	A.coeffRef(12, 6) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(12, 12) = dcomplex(-0.5 * Gamma_31 - gamma_c - gamma_p, 1.0 * (-delta_p - k_p * v - (Delta_23 + delta_c - k_c * v)));
	A.coeffRef(12, 13) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(13, 0) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(13, 2) = dcomplex(0, 0.5 * Omega_13);
	A.coeffRef(13, 12) = dcomplex(0, 0.5 * Omega_01);
	A.coeffRef(13, 13) = dcomplex(-0.5 * Gamma_10 - 0.5 * Gamma_31 - gamma_c, -1.0 * (Delta_23 + delta_c - k_c * v));
	A.coeffRef(13, 14) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(14, 7) = dcomplex(0, -0.5 * Omega_13);
	A.coeffRef(14, 13) = dcomplex(0, 0.5 * Omega_12);
	A.coeffRef(14, 14) = dcomplex(-0.5 * Gamma_21 - 0.5 * Gamma_31, 1.0 * (delta_c - k_c * v - (Delta_23 + delta_c - k_c * v)));

	A.makeCompressed();

	return A;
}



int main() {

	std::cout << "Probe Laser Wavelength " << std::endl;
	std::cin >> lambdaP;
	lambdaP = lambdaP * 1e-07;

	std::cout << "Coupling Laser Wavelength" << std::endl;
	std::cin >> lambdaC;
	lambdaC = lambdaC * 1e-07;

	alphaP_0 = (3 / (2 * M_PI)) * pow(lambdaP, 2) * density;
	alphaC_0 = (3 / (2 * M_PI)) * pow(lambdaC, 2) * density;

	std::cout << "alphaP_0 = " << alphaP_0 << "cm^-1" << std::endl;
	std::cout << "alphaC_0 = " << alphaC_0 << "cm^-1" << std::endl;

	k_p = 1 / lambdaP * 1e-06;
	k_c = 1 / lambdaC * 1e-06;

	std::cout << "k_P = " << k_p << "cm^-1" << std::endl;
	std::cout << "k_C = " << k_c << "cm^-1" << std::endl;

	std::cout << "Omega_01" << std::endl;
	std::cin >> Omega_01;

	std::cout << "Omega_12" << std::endl;
	std::cin >> Omega_12;

	std::cout << "Omega_13" << std::endl;
	std::cin >> Omega_13;


	std::cout << "Gamma_21" << std::endl;
	std::cin >> Gamma_21;

	std::cout << "Gamma_31" << std::endl;
	std::cin >> Gamma_31;

	std::cout << "Gamma_10" << std::endl;
	std::cin >> Gamma_10;

	std::cout << "Probe Laser Linewidth" << std::endl;
	std::cin >> gamma_p;

	std::cout << "Coupling Laser Linewidth" << std::endl;
	std::cin >> gamma_c;

	std::cout << "Probe Laser Detuning" << std::endl;
	std::cin >> delta_p;

	std::cout << "Hyperfine Splitting" << std::endl;
	std::cin >> Delta_23;

	std::cout << "Initial Detuning" << std::endl;
	std::cin >> initialDetuning;

	std::cout << "Final Detuning" << std::endl;
	std::cin >> finalDetuning;

	std::cout << "Number of Detuning Points" << std::endl;
	std::cin >> numberDetunings;

	std::cout << "Initial Velocity" << std::endl;
	std::cin >> initialVelocity;

	std::cout << "Final Velocity" << std::endl;
	std::cin >> finalVelocity;

	std::cout << "Number of Velocity Points" << std::endl;
	std::cin >> numberVelocity;

	std::cout << "Outpute File Name" << std::endl;
	std::cin >> fileName;

	fileName = fileName + ".csv";

	double dv;
	double dc;
	double vAtom;
	double coupling;
	double weight;

	std::ofstream myFile;

	myFile.open(fileName);
	
	myFile << "Coupling Detunings,TransmissionP_D \n";

	SparseVector<dcomplex> alphaP_D(numberDetunings);

	dv = (finalVelocity - initialVelocity) / (numberVelocity - 1);
	dc = (finalDetuning - initialDetuning) / (numberDetunings - 1);

	b.coeffRef(3) = dcomplex(0, -0.5*Omega_01);
	b.coeffRef(6) = dcomplex(0, 0.5*Omega_01);

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < numberDetunings; i++) {

		coupling = initialDetuning + i * dc;

		for (int k = 0; k < numberVelocity; k++) {

			vAtom = initialVelocity + k * dv;

			weight = 1 / (sqrt(M_PI) * u) * exp(-1.0 * (pow(vAtom, 2) / pow(u, 2))) * dv;

			B = M(coupling, vAtom);

			solver.analyzePattern(B);
			solver.factorize(B);

			x = solver.solve(b);

			alphaP_D.coeffRef(i) += weight * alphaP_0 * (Gamma_10 / Omega_01) * ((dcomplex(0, 1) * x.coeffRef(6)).real()); // x.coeffRef(6) is rho_10

		}

		myFile << coupling << "," << exp(-cellLength * alphaP_D.coeffRef(i)).real() << "\n";
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	std::cout << "Function Runtime : " << duration.count() << std::endl;

	std::system("pause");

}
